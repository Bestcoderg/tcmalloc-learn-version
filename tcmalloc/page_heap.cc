// Copyright 2019 The TCMalloc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tcmalloc/page_heap.h"

#include <stddef.h>

#include <limits>

#include "absl/base/internal/cycleclock.h"
#include "absl/base/internal/spinlock.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/page_heap_allocator.h"
#include "tcmalloc/pagemap.h"
#include "tcmalloc/parameters.h"
#include "tcmalloc/static_vars.h"
#include "tcmalloc/system-alloc.h"

namespace tcmalloc {

// Helper function to record span address into pageheap
void PageHeap::RecordSpan(Span* span) {
  pagemap_->Set(span->first_page(), span);
  if (span->num_pages() > 1) {
    pagemap_->Set(span->last_page(), span);
  }
}

PageHeap::PageHeap(bool tagged) : PageHeap(Static::pagemap(), tagged) {}

PageHeap::PageHeap(PageMap* map, bool tagged)
    : PageAllocatorInterface("PageHeap", map, tagged),
      scavenge_counter_(0),
      // Start scavenging at kMaxPages list
      release_index_(kMaxPages) {
  large_.normal.Init();
  large_.returned.Init();
  for (int i = 0; i < kMaxPages; i++) {
    free_[i].normal.Init();
    free_[i].returned.Init();
  }
}

Span* PageHeap::SearchFreeAndLargeLists(Length n, bool* from_returned) {
  ASSERT(Check());
  ASSERT(n > 0);

  // Find first size >= n that has a non-empty list
  for (Length s = n; s < kMaxPages; s++) {
    SpanList* ll = &free_[s].normal;
    // If we're lucky, ll is non-empty, meaning it has a suitable span.
    if (!ll->empty()) {
      ASSERT(ll->first()->location() == Span::ON_NORMAL_FREELIST);
      *from_returned = false;
      // 将此Span分割出长度为n的Span
      return Carve(ll->first(), n);
    }
    // Alternatively, maybe there's a usable returned span.
    ll = &free_[s].returned;
    if (!ll->empty()) {
      ASSERT(ll->first()->location() == Span::ON_RETURNED_FREELIST);
      *from_returned = true;
      return Carve(ll->first(), n);
    }
  }
  // No luck in free lists, our last chance is in a larger class.
  return AllocLarge(n, from_returned);  // May be NULL
}

Span* PageHeap::AllocateSpan(Length n, bool* from_returned) {
  ASSERT(Check());
  Span* result = SearchFreeAndLargeLists(n, from_returned);
  if (result != nullptr) return result;

  // Grow the heap and try again.
  if (!GrowHeap(n)) {
    ASSERT(Check());
    return nullptr;
  }

  result = SearchFreeAndLargeLists(n, from_returned);
  // our new memory should be unbacked
  ASSERT(*from_returned);
  return result;
}

Span* PageHeap::New(Length n) {
  ASSERT(n > 0);
  bool from_returned;
  Span* result;
  {
    // 全局锁
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    // Span 连续空间
    result = AllocateSpan(n, &from_returned);
    // 是否需要释放空闲的 Span/Pages 到 system
    if (result) Static::page_allocator()->ShrinkToUsageLimit();
    if (result) info_.RecordAlloc(result->first_page(), result->num_pages());
  }

  if (result != nullptr && from_returned) {
    SystemBack(result->start_address(), result->bytes_in_span());
  }

  ASSERT(!result || IsTaggedMemory(result->start_address()) == tagged_);
  return result;
}

static bool IsSpanBetter(Span* span, Span* best, Length n) {
// 优先取较小的page,大小相同则取位置靠前的page
  if (span->num_pages() < n) {
    return false;
  }
  if (best == nullptr) {
    return true;
  }
  if (span->num_pages() < best->num_pages()) {
    return true;
  }
  if (span->num_pages() > best->num_pages()) {
    return false;
  }
  return span->first_page() < best->first_page();
}

// We could do slightly more efficient things here (we do some
// unnecessary Carves in New) but it's not anywhere
// close to a fast path, and is going to be replaced soon anyway, so
// don't bother.
Span* PageHeap::NewAligned(Length n, Length align) {
  ASSERT(n > 0);
  ASSERT((align & (align - 1)) == 0);

  if (align <= 1) {
    return New(n);
  }

  bool from_returned;
  Span* span;
  {
    absl::base_internal::SpinLockHolder h(&pageheap_lock);
    Length extra = align - 1;
    span = AllocateSpan(n + extra, &from_returned);
    if (span == nullptr) return nullptr;
    // <span> certainly contains an appropriately aligned region; find it
    // and chop off the rest.
    PageID p = span->first_page();
    const PageID mask = align - 1;
    PageID aligned = (p + mask) & ~mask;
    ASSERT(aligned % align == 0);
    ASSERT(p <= aligned);
    ASSERT(aligned + n <= p + span->num_pages());
    // we have <extra> too many pages now, possible all before, possibly all
    // after, maybe both
    Length before = aligned - p;
    Length after = extra - before;
    span->set_first_page(aligned);
    span->set_num_pages(n);
    RecordSpan(span);

    const Span::Location loc =
        from_returned ? Span::ON_RETURNED_FREELIST : Span::ON_NORMAL_FREELIST;
    if (before > 0) {
      Span* extra = Span::New(p, before);
      extra->set_location(loc);
      RecordSpan(extra);
      MergeIntoFreeList(extra);
    }

    if (after > 0) {
      Span* extra = Span::New(aligned + n, after);
      extra->set_location(loc);
      RecordSpan(extra);
      MergeIntoFreeList(extra);
    }

    info_.RecordAlloc(aligned, n);
  }

  if (span != nullptr && from_returned) {
    SystemBack(span->start_address(), span->bytes_in_span());
  }

  ASSERT(!span || IsTaggedMemory(span->start_address()) == tagged_);
  return span;
}

Span* PageHeap::AllocLarge(Length n, bool* from_returned) {
  // find the best span (closest to n in size).
  // The following loops implements address-ordered best-fit.
  Span* best = nullptr;

  // Search through normal list
  for (Span* span : large_.normal) {
    ASSERT(span->location() == Span::ON_NORMAL_FREELIST);
    if (IsSpanBetter(span, best, n)) {
      best = span;
      *from_returned = false;
    }
  }

  // Search through released list in case it has a better fit
  for (Span* span : large_.returned) {
    ASSERT(span->location() == Span::ON_RETURNED_FREELIST);
    if (IsSpanBetter(span, best, n)) {
      best = span;
      *from_returned = true;
    }
  }

  return best == nullptr ? nullptr : Carve(best, n);
}

Span* PageHeap::Carve(Span* span, Length n) {
  ASSERT(n > 0);
  ASSERT(span->location() != Span::IN_USE);
  const Span::Location old_location = span->location();
  RemoveFromFreeList(span);
  span->set_location(Span::IN_USE);

  const int extra = span->num_pages() - n;
  ASSERT(extra >= 0);
  if (extra > 0) {
    Span* leftover = nullptr;
    // Check if this span has another span on the right but not on the left.
    // There is one special case we want to handle: if heap grows down (as it is
    // usually happens with mmap allocator) and user allocates lots of large
    // persistent memory blocks (namely, kMinSystemAlloc + epsilon), then we
    // want to return the last part of the span to user and push the beginning
    // to the freelist.
    // Otherwise system allocator would allocate 2 * kMinSystemAlloc, we return
    // the first kMinSystemAlloc + epsilon to user and add the remaining
    // kMinSystemAlloc - epsilon to the freelist. The remainder is not large
    // enough to satisfy the next allocation request, so we allocate
    // another 2 * kMinSystemAlloc from system and the process repeats wasting
    // half of memory.
    // If we return the last part to user, then the remainder will be merged
    // with the next system allocation which will result in dense packing.
    // There are no other known cases where span splitting strategy matters,
    // so in other cases we return beginning to user.
    if (pagemap_->GetDescriptor(span->first_page() - 1) == nullptr &&
        pagemap_->GetDescriptor(span->last_page() + 1) != nullptr) {
      leftover = Span::New(span->first_page(), extra);
      span->set_first_page(span->first_page() + extra);
      pagemap_->Set(span->first_page(), span);
    } else {
      leftover = Span::New(span->first_page() + n, extra);
    }
    leftover->set_location(old_location);
    RecordSpan(leftover);
    PrependToFreeList(leftover);  // Skip coalescing - no candidates possible
    leftover->set_freelist_added_time(span->freelist_added_time());
    span->set_num_pages(n);
    pagemap_->Set(span->last_page(), span);
  }
  ASSERT(Check());
  return span;
}

void PageHeap::Delete(Span* span) {
  ASSERT(IsTaggedMemory(span->start_address()) == tagged_);
  info_.RecordFree(span->first_page(), span->num_pages());
  ASSERT(Check());
  ASSERT(span->location() == Span::IN_USE);
  ASSERT(!span->sampled());
  ASSERT(span->num_pages() > 0);
  ASSERT(pagemap_->GetDescriptor(span->first_page()) == span);
  ASSERT(pagemap_->GetDescriptor(span->last_page()) == span);
  const Length n = span->num_pages();
  span->set_location(Span::ON_NORMAL_FREELIST);
  MergeIntoFreeList(span);  // Coalesces if possible
  IncrementalScavenge(n);
  ASSERT(Check());
}

void PageHeap::MergeIntoFreeList(Span* span) {
  ASSERT(span->location() != Span::IN_USE);
  span->set_freelist_added_time(absl::base_internal::CycleClock::Now());
 
  // Coalesce -- we guarantee that "p" != 0, so no bounds checking
  // necessary.  We do not bother resetting the stale pagemap
  // entries for the pieces we are merging together because we only
  // care about the pagemap entries for the boundaries.
  //
  // Note that only similar spans are merged together.  For example,
  // we do not coalesce "returned" spans with "normal" spans.
  // 
  // 如果前后有能够合并的Span,则将前后的span合并到这个span中,减少维护的span数量
  const PageID p = span->first_page();
  const Length n = span->num_pages();
  Span* prev = pagemap_->GetDescriptor(p - 1);
  if (prev != nullptr && prev->location() == span->location()) {
    // Merge preceding span into this span
    ASSERT(prev->last_page() + 1 == p);
    const Length len = prev->num_pages();
    span->AverageFreelistAddedTime(prev);
    RemoveFromFreeList(prev);
    Span::Delete(prev);
    span->set_first_page(span->first_page() - len);
    span->set_num_pages(span->num_pages() + len);
    pagemap_->Set(span->first_page(), span);
  }
  Span* next = pagemap_->GetDescriptor(p + n);
  if (next != nullptr && next->location() == span->location()) {
    // Merge next span into this span
    ASSERT(next->first_page() == p + n);
    const Length len = next->num_pages();
    span->AverageFreelistAddedTime(next);
    RemoveFromFreeList(next);
    Span::Delete(next);
    span->set_num_pages(span->num_pages() + len);
    pagemap_->Set(span->last_page(), span);
  }

  PrependToFreeList(span);
}

void PageHeap::PrependToFreeList(Span* span) {
  ASSERT(span->location() != Span::IN_USE);
  SpanListPair* list =
      (span->num_pages() < kMaxPages) ? &free_[span->num_pages()] : &large_;
  if (span->location() == Span::ON_NORMAL_FREELIST) {
    stats_.free_bytes += span->bytes_in_span();
    list->normal.prepend(span);
  } else {
    stats_.unmapped_bytes += span->bytes_in_span();
    list->returned.prepend(span);
  }
}

void PageHeap::RemoveFromFreeList(Span* span) {
  ASSERT(span->location() != Span::IN_USE);
  if (span->location() == Span::ON_NORMAL_FREELIST) {
    stats_.free_bytes -= span->bytes_in_span();
  } else {
    stats_.unmapped_bytes -= span->bytes_in_span();
  }
  span->RemoveFromList();
}

void PageHeap::IncrementalScavenge(Length n) {
}

Length PageHeap::ReleaseLastNormalSpan(SpanListPair* slist) {
  Span* s = slist->normal.last();
  ASSERT(s->location() == Span::ON_NORMAL_FREELIST);
  RemoveFromFreeList(s);

  // We're dropping very important and otherwise contended pageheap_lock around
  // call to potentially very slow syscall to release pages. Those syscalls can
  // be slow even with "advanced" things such as MADV_FREE{,ABLE} because they
  // have to walk actual page tables, and we sometimes deal with large spans,
  // which sometimes takes lots of time. Plus Linux grabs per-address space
  // mm_sem lock which could be extremely contended at times. So it is best if
  // we avoid holding one contended lock while waiting for another.
  //
  // Note, we set span location to in-use, because our span could be found via
  // pagemap in e.g. MergeIntoFreeList while we're not holding the lock. By
  // marking it in-use we prevent this possibility. So span is removed from free
  // list and marked "unmergable" and that guarantees safety during unlock-ful
  // release.
  //
  // Taking the span off the free list will make our stats reporting wrong if
  // another thread happens to try to measure memory usage during the release,
  // so we fix up the stats during the unlocked period.
  stats_.free_bytes += s->bytes_in_span();
  s->set_location(Span::IN_USE);
  pageheap_lock.Unlock();

  // 将内存释放给系统
  const Length n = s->num_pages();
  SystemRelease(s->start_address(), s->bytes_in_span());

  pageheap_lock.Lock();
  stats_.free_bytes -= s->bytes_in_span();
  s->set_location(Span::ON_RETURNED_FREELIST);
  MergeIntoFreeList(s);  // Coalesces if possible.
  return n;
}

Length PageHeap::ReleaseAtLeastNPages(Length num_pages) {
  Length released_pages = 0;
  Length prev_released_pages = -1;

  // Round robin through the lists of free spans, releasing the last
  // span in each list.  Stop after releasing at least num_pages.
  // 循环遍历空的span list,每次释放最后一个span
  while (released_pages < num_pages) {
    if (released_pages == prev_released_pages) {
      // Last iteration of while loop made no progress.
      break;
    }
    prev_released_pages = released_pages;

    for (int i = 0; i < kMaxPages+1 && released_pages < num_pages;
         i++, release_index_++) {
      // 扫描一遍不同大小的spans
      if (release_index_ > kMaxPages) release_index_ = 0;
      SpanListPair* slist =
          (release_index_ == kMaxPages) ? &large_ : &free_[release_index_];
      if (!slist->normal.empty()) {
        Length released_len = ReleaseLastNormalSpan(slist);
        released_pages += released_len;
      }
    }
  }
  info_.RecordRelease(num_pages, released_pages);
  return released_pages;
}

void PageHeap::GetSmallSpanStats(SmallSpanStats* result) {
  for (int s = 0; s < kMaxPages; s++) {
    result->normal_length[s] = free_[s].normal.length();
    result->returned_length[s] = free_[s].returned.length();
  }
}

void PageHeap::GetLargeSpanStats(LargeSpanStats* result) {
  result->spans = 0;
  result->normal_pages = 0;
  result->returned_pages = 0;
  for (Span* s : large_.normal) {
    result->normal_pages += s->num_pages();
    result->spans++;
  }
  for (Span* s : large_.returned) {
    result->returned_pages += s->num_pages();
    result->spans++;
  }
}

bool PageHeap::GrowHeap(Length n) {
  if (n > kMaxValidPages) return false;
  size_t actual_size;
  void* ptr = SystemAlloc(n << kPageShift, &actual_size, kPageSize, tagged_);
  if (ptr == nullptr) return false;
  n = actual_size >> kPageShift;

  stats_.system_bytes += actual_size;
  // addr => PageID 将addr转为唯一PageID
  // 这样就可以轻松知道 前后的 page 的状态了
  const PageID p = reinterpret_cast<uintptr_t>(ptr) >> kPageShift;
  ASSERT(p > 0);

  // If we have already a lot of pages allocated, just pre allocate a bunch of
  // memory for the page map. This prevents fragmentation by pagemap metadata
  // when a program keeps allocating and freeing large blocks.

  // Make sure pagemap has entries for all of the new pages.
  // Plus ensure one before and one after so coalescing code
  // does not need bounds-checking.
  // 确保申请的这片内存用于维护的Leaf的存在,维护一下PageMap
  if (pagemap_->Ensure(p - 1, n + 2)) {
    // Pretend the new area is allocated and then return it to cause
    // any necessary coalescing to occur.
    Span* span = Span::New(p, n);
    // 将span加入pagemap(Leaf)维护
    RecordSpan(span);
    span->set_location(Span::ON_RETURNED_FREELIST);
    // 将这个新申请的span进行合并与加入freelist的操作
    MergeIntoFreeList(span);
    ASSERT(Check());
    return true;
  } else {
    // We could not allocate memory within the pagemap.
    // Note the following leaks virtual memory, but at least it gets rid of
    // the underlying physical memory.
    SystemRelease(ptr, actual_size);
    return false;
  }
}

bool PageHeap::Check() {
  ASSERT(free_[0].normal.empty());
  ASSERT(free_[0].returned.empty());
  return true;
}

void PageHeap::PrintInPbtxt(PbtxtRegion* region) {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  SmallSpanStats small;
  GetSmallSpanStats(&small);
  LargeSpanStats large;
  GetLargeSpanStats(&large);

  struct Helper {
    static void RecordAges(PageAgeHistograms* ages, const SpanListPair& pair) {
      for (const Span* s : pair.normal) {
        ages->RecordRange(s->num_pages(), false, s->freelist_added_time());
      }

      for (const Span* s : pair.returned) {
        ages->RecordRange(s->num_pages(), true, s->freelist_added_time());
      }
    }
  };

  PageAgeHistograms ages(absl::base_internal::CycleClock::Now());
  for (int s = 0; s < kMaxPages; ++s) {
    Helper::RecordAges(&ages, free_[s]);
  }
  Helper::RecordAges(&ages, large_);
  PrintStatsInPbtxt(region, small, large, ages);
  // We do not collect info_.PrintInPbtxt for now.
}

void PageHeap::Print(TCMalloc_Printer* out) {
  absl::base_internal::SpinLockHolder h(&pageheap_lock);
  SmallSpanStats small;
  GetSmallSpanStats(&small);
  LargeSpanStats large;
  GetLargeSpanStats(&large);
  PrintStats("PageHeap", out, stats_, small, large, true);

  struct Helper {
    static void RecordAges(PageAgeHistograms* ages, const SpanListPair& pair) {
      for (const Span* s : pair.normal) {
        ages->RecordRange(s->num_pages(), false, s->freelist_added_time());
      }

      for (const Span* s : pair.returned) {
        ages->RecordRange(s->num_pages(), true, s->freelist_added_time());
      }
    }
  };

  PageAgeHistograms ages(absl::base_internal::CycleClock::Now());
  for (int s = 0; s < kMaxPages; ++s) {
    Helper::RecordAges(&ages, free_[s]);
  }
  Helper::RecordAges(&ages, large_);
  ages.Print("PageHeap", out);

  info_.Print(out);
}

}  // namespace tcmalloc
