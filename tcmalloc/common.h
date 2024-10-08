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
//
// Common definitions for tcmalloc code.

#ifndef TCMALLOC_COMMON_H_
#define TCMALLOC_COMMON_H_

#include <bits/wordsize.h>
#include <stddef.h>
#include <stdint.h>

#include "absl/base/attributes.h"
#include "absl/base/internal/spinlock.h"
#include "absl/base/optimization.h"
#include "tcmalloc/internal/bits.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/size_class_info.h"

// Type that can hold a page number
typedef uintptr_t PageID;

// Type that can hold the length of a run of pages
typedef uintptr_t Length;

//-------------------------------------------------------------------
// Configuration
//-------------------------------------------------------------------

// There are four different models for tcmalloc which are created by defining a
// set of constant variables differently:
//
// DEFAULT:
//   The default configuration strives for good performance while trying to
//   minimize fragmentation.  It uses a smaller page size to reduce
//   fragmentation, but allocates per-thread and per-cpu capacities similar to
//   TCMALLOC_LARGE_PAGES / TCMALLOC_256K_PAGES.
//
// TCMALLOC_LARGE_PAGES:
//   Larger page sizes increase the bookkeeping granularity used by TCMalloc for
//   its allocations.  This can reduce PageMap size and traffic to the
//   innermost cache (the page heap), but can increase memory footprints.  As
//   TCMalloc will not reuse a page for a different allocation size until the
//   entire page is deallocated, this can be a source of increased memory
//   fragmentation.
//
//   Historically, larger page sizes improved lookup performance for the
//   pointer-to-size lookup in the PageMap that was part of the critical path.
//   With most deallocations leveraging C++14's sized delete feature
//   (https://isocpp.org/files/papers/n3778.html), this optimization is less
//   significant.
//
// TCMALLOC_256K_PAGES
//   This configuration uses an even larger page size (256KB) as the unit of
//   accounting granularity.
//
// TCMALLOC_SMALL_BUT_SLOW:
//   Used for situations where minimizing the memory footprint is the most
//   desirable attribute, even at the cost of performance.
//
// The constants that vary between models are:
//
//   kPageShift - Shift amount used to compute the page size.
//   kNumClasses - Number of size classes serviced by bucket allocators
//   kMaxSize - Maximum size serviced by bucket allocators (thread/cpu/central)
//   kMinThreadCacheSize - The minimum size in bytes of each ThreadCache.
//   kMaxThreadCacheSize - The maximum size in bytes of each ThreadCache.
//   kDefaultOverallThreadCacheSize - The maximum combined size in bytes of all
//     ThreadCaches for an executable.
//   kStealAmount - The number of bytes one ThreadCache will steal from another
//     when the first ThreadCache is forced to Scavenge(), delaying the next
//     call to Scavenge for this thread.

// Older configurations had their own customized macros.  Convert them into
// a page-shift parameter that is checked below.

#ifndef TCMALLOC_PAGE_SHIFT
#ifdef TCMALLOC_SMALL_BUT_SLOW
#define TCMALLOC_PAGE_SHIFT 12
#define TCMALLOC_USE_PAGEMAP3
#elif defined(TCMALLOC_256K_PAGES)
#define TCMALLOC_PAGE_SHIFT 18
#elif defined(TCMALLOC_LARGE_PAGES)
#define TCMALLOC_PAGE_SHIFT 15
#else
#define TCMALLOC_PAGE_SHIFT 13
#endif
#else
#error "TCMALLOC_PAGE_SHIFT is an internal macro!"
#endif

#if TCMALLOC_PAGE_SHIFT == 12
static const size_t kPageShift = 12;
static const size_t kNumClasses = 46;
static const size_t kMaxSize = 8 << 10;
static const size_t kMinThreadCacheSize = 4 * 1024;
static const size_t kMaxThreadCacheSize = 64 * 1024;
static const size_t kMaxCpuCacheSize = 20 * 1024;
static const size_t kDefaultOverallThreadCacheSize = kMaxThreadCacheSize;
static const size_t kStealAmount = kMinThreadCacheSize;
static const size_t kDefaultProfileSamplingRate = 1 << 19;
static const size_t kMinPages = 2;
#elif TCMALLOC_PAGE_SHIFT == 15
static const size_t kPageShift = 15;
static const size_t kNumClasses = 78;
static const size_t kMaxSize = 256 * 1024;
static const size_t kMinThreadCacheSize = kMaxSize * 2;
static const size_t kMaxThreadCacheSize = 4 << 20;
static const size_t kMaxCpuCacheSize = 3 * 1024 * 1024;
static const size_t kDefaultOverallThreadCacheSize = 8u * kMaxThreadCacheSize;
static const size_t kStealAmount = 1 << 16;
static const size_t kDefaultProfileSamplingRate = 1 << 21;
static const size_t kMinPages = 8;
#elif TCMALLOC_PAGE_SHIFT == 18
static const size_t kPageShift = 18;
static const size_t kNumClasses = 89;
static const size_t kMaxSize = 256 * 1024;
static const size_t kMinThreadCacheSize = kMaxSize * 2;
static const size_t kMaxThreadCacheSize = 4 << 20;
static const size_t kMaxCpuCacheSize = 3 * 1024 * 1024;
static const size_t kDefaultOverallThreadCacheSize = 8u * kMaxThreadCacheSize;
static const size_t kStealAmount = 1 << 16;
static const size_t kDefaultProfileSamplingRate = 1 << 21;
static const size_t kMinPages = 8;
#elif TCMALLOC_PAGE_SHIFT == 13
// 8K
static const size_t kPageShift = 13;
static const size_t kNumClasses = 86;
static const size_t kMaxSize = 256 * 1024;
static const size_t kMinThreadCacheSize = kMaxSize * 2;
static const size_t kMaxThreadCacheSize = 4 << 20;
static const size_t kMaxCpuCacheSize = 3 * 1024 * 1024;
static const size_t kDefaultOverallThreadCacheSize = 8u * kMaxThreadCacheSize;
static const size_t kStealAmount = 1 << 16;
static const size_t kDefaultProfileSamplingRate = 1 << 21;
static const size_t kMinPages = 8;
#else
#error "Unsupported TCMALLOC_PAGE_SHIFT value!"
#endif

// Minimum/maximum number of batches in TransferCache per size class.
// Actual numbers depends on a number of factors, see TransferCache::Init
// for details.
static const size_t kMinObjectsToMove = 2;
static const size_t kMaxObjectsToMove = 128;

static const size_t kPageSize   = 1 << kPageShift;
// Verify that the page size used is at least 8x smaller than the maximum
// element size in the thread cache.  This guarantees at most 12.5% internal
// fragmentation (1/8). When page size is 256k (kPageShift == 18), the benefit
// of increasing kMaxSize to be multiple of kPageSize is unclear. Object size
// profile data indicates that the number of simultaneously live objects (of
// size >= 256k) tends to be very small. Keeping those objects as 'large'
// objects won't cause too much memory waste, while heap memory reuse is can be
// improved. Increasing kMaxSize to be too large has another bad side effect --
// the thread cache pressure is increased, which will in turn increase traffic
// between central cache and thread cache, leading to performance degradation.
static_assert((kMaxSize / kPageSize) >= kMinPages || kPageShift >= 18,
              "Ratio of kMaxSize / kPageSize is too small");

static const size_t kAlignment  = 8;
// log2 (kAlignment)
static const size_t kAlignmentShift =
    tcmalloc::tcmalloc_internal::Bits::Log2Ceiling(kAlignment);
// For all span-lengths < kMaxPages we keep an exact-size list.
static const size_t kMaxPages = 1 << (20 - kPageShift);

// The number of times that a deallocation can cause a freelist to
// go over its max_length() before shrinking max_length().
static const int kMaxOverages = 3;

// Maximum length we allow a per-thread free-list to have before we
// move objects from it into the corresponding central free-list.  We
// want this big to avoid locking the central free-list too often.  It
// should not hurt to make this list somewhat big because the
// scavenging code will shrink it down when its contents are not in use.
static const int kMaxDynamicFreeListLength = 8192;

static const Length kMaxValidPages = (~static_cast<Length>(0)) >> kPageShift;

#if defined __x86_64__
// All current and planned x86_64 processors only look at the lower 48 bits
// in virtual to physical address translation.  The top 16 are thus unused.
// TODO(b/134686025): Under what operating systems can we increase it safely to
// 17? This lets us use smaller page maps.  On first allocation, a 36-bit page
// map uses only 96 KB instead of the 4.5 MB used by a 52-bit page map.
static const int kAddressBits = (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 48);
#elif defined __powerpc64__ && defined __linux__
// Linux(4.12 and above) on powerpc64 supports 128TB user virtual address space
// by default, and up to 512TB if user space opts in by specifing hint in mmap.
// See comments in arch/powerpc/include/asm/processor.h
// and arch/powerpc/mm/mmap.c.
static const int kAddressBits = (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 49);
#elif defined __aarch64__ && defined __linux__
// According to Documentation/arm64/memory.txt of kernel 3.16,
// AARCH64 kernel supports 48-bit virtual addresses for both user and kernel.
static const int kAddressBits = (sizeof(void*) < 8 ? (8 * sizeof(void*)) : 48);
#else
static const int kAddressBits = 8 * sizeof(void*);
#endif

namespace tcmalloc {
#if defined(__x86_64__)
// x86 has 2 MiB huge pages
static const size_t kHugePageShift = 21;
#elif defined(__PPC64__)
static const size_t kHugePageShift = 24;
#elif defined __aarch64__ && defined __linux__
static const size_t kHugePageShift = 21;
#else
// ...whatever, guess something big-ish
static const size_t kHugePageShift = 21;
#endif

static const size_t kHugePageSize = static_cast<size_t>(1) << kHugePageShift;
static const size_t kPagesPerHugePage = static_cast<size_t>(1)
                                        << (kHugePageShift - kPageShift);
static constexpr uintptr_t kTagMask = uintptr_t{1}
                                      << std::min(kAddressBits - 4, 42);

#if !defined(TCMALLOC_SMALL_BUT_SLOW) && __WORDSIZE != 32
// Always allocate at least a huge page
static const size_t kMinSystemAlloc = kHugePageSize;
static const size_t kMinMmapAlloc = 1 << 30;  // mmap() in 1GiB ranges.
#else
// Allocate in units of 2MiB. This is the size of a huge page for x86, but
// not for Power.
static const size_t kMinSystemAlloc = 2 << 20;
// mmap() in units of 32MiB. This is a multiple of huge page size for
// both x86 (2MiB) and Power (16MiB)
static const size_t kMinMmapAlloc = 32 << 20;
#endif

static_assert(kMinMmapAlloc % kMinSystemAlloc == 0,
              "Minimum mmap allocation size is not a multiple of"
              " minimum system allocation size");

// Convert byte size into pages.  This won't overflow, but may return
// an unreasonably large value if bytes is huge enough.
inline Length pages(size_t bytes) {
  return (bytes >> kPageShift) +
      ((bytes & (kPageSize - 1)) > 0 ? 1 : 0);
}

// Returns true if ptr is tagged.
inline bool IsTaggedMemory(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & kTagMask) == 0;
}

// Size-class information + mapping
class SizeMap {
 public:
  // All size classes <= 512 in all configs always have 1 page spans.
  static const size_t kMultiPageSize = 512;
  // Min alignment for all size classes > kMultiPageSize in all configs.
  static const size_t kMultiPageAlignment = 64;
  // log2 (kMultiPageAlignment)
  static const size_t kMultiPageAlignmentShift =
      tcmalloc::tcmalloc_internal::Bits::Log2Ceiling(kMultiPageAlignment);

 private:
  //-------------------------------------------------------------------
  // Mapping from size to size_class and vice versa
  //-------------------------------------------------------------------

  // Sizes <= 1024 have an alignment >= 8.  So for such sizes we have an
  // array indexed by ceil(size/8).  Sizes > 1024 have an alignment >= 128.
  // So for these larger sizes we have an array indexed by ceil(size/128).
  //
  // We flatten both logical arrays into one physical array and use
  // arithmetic to compute an appropriate index.  The constants used by
  // ClassIndex() were selected to make the flattening work.
  //
  // Examples:
  //   Size       Expression                      Index
  //   -------------------------------------------------------
  //   0          (0 + 7) / 8                     0
  //   1          (1 + 7) / 8                     1
  //   ...
  //   1024       (1024 + 7) / 8                  128
  //   1025       (1025 + 127 + (120<<7)) / 128   129
  //   ...
  //   32768      (32768 + 127 + (120<<7)) / 128  376
  static const int kMaxSmallSize = 1024;
  static const size_t kClassArraySize =
      ((kMaxSize + 127 + (120 << 7)) >> 7) + 1;

  // Batch size is the number of objects to move at once.
  typedef unsigned char BatchSize;

  // class_array_ is accessed on every malloc, so is very hot.  We make it the
  // first member so that it inherits the overall alignment of a SizeMap
  // instance.  In particular, if we create a SizeMap instance that's cache-line
  // aligned, this member is also aligned to the width of a cache line.
  // 索引到 size-class index, 相当于一个映射表,为了快速找到对应的size-class(我们很难直接使用size 直接找到 size-class)
  unsigned char class_array_[kClassArraySize];

  // Number of objects to move between a per-thread list and a central
  // list in one shot.  We want this to be not too small so we can
  // amortize the lock overhead for accessing the central list.  Making
  // it too big may temporarily cause unnecessary memory wastage in the
  // per-thread free list until the scavenger cleans up the list.
  BatchSize num_objects_to_move_[kNumClasses];

  // If size is no more than kMaxSize, compute index of the
  // class_array[] entry for it, putting the class index in output
  // parameter idx and returning true. Otherwise return false.
  static inline bool ABSL_ATTRIBUTE_ALWAYS_INLINE ClassIndexMaybe(size_t s,
                                                                  uint32_t* idx) {
    if (ABSL_PREDICT_TRUE(s <= kMaxSmallSize)) {
      *idx = (static_cast<uint32_t>(s) + 7) >> 3;
      return true;
    } else if (s <= kMaxSize) {
      *idx = (static_cast<uint32_t>(s) + 127 + (120 << 7)) >> 7;
      return true;
    }
    return false;
  }

  static inline size_t ClassIndex(size_t s) {
    uint32_t ret;
    CHECK_CONDITION(ClassIndexMaybe(s, &ret));
    return ret;
  }

  // Mapping from size class to number of pages to allocate at a time
  unsigned char class_to_pages_[kNumClasses];

  // Mapping from size class to max size storable in that class
  uint32_t class_to_size_[kNumClasses];

  // If environment variable defined, use it to override sizes classes.
  // Returns true if all classes defined correctly.
  bool MaybeRunTimeSizeClasses();

 protected:
  // Set the give size classes to be used by TCMalloc.
  void SetSizeClasses(int num_classes, const SizeClassInfo* parsed);

  // Check that the size classes meet all requirements.
  bool ValidSizeClasses(int num_classes, const SizeClassInfo* parsed);

  // Definition of size class that is set in size_classes.cc
  static const SizeClassInfo kSizeClasses[kNumClasses];

  // Definition of size class that is set in size_classes.cc
  static const SizeClassInfo kExperimentalSizeClasses[kNumClasses];

 public:
  // Constructor should do nothing since we rely on explicit Init()
  // call, which may or may not be called before the constructor runs.
  SizeMap() { }

  // Initialize the mapping arrays
  void Init();

  // Returns the non-zero matching size class for the provided `size`.
  // Returns true on success, returns false if `size` exceeds the maximum size
  // class value `kMaxSize'.
  // Important: this function may return true with *cl == 0 if this
  // SizeMap instance has not (yet) been initialized.
  inline bool ABSL_ATTRIBUTE_ALWAYS_INLINE GetSizeClass(size_t size,
                                                        uint32_t* cl) {
    uint32_t idx;
    if (ABSL_PREDICT_TRUE(ClassIndexMaybe(size, &idx))) {
      *cl = class_array_[idx];
      return true;
    }
    return false;
  }

  // Returns the size class for size `size` aligned at `align`
  // Returns true on success. Returns false if either:
  // - the size exceeds the maximum size class size.
  // - the align size is greater or equal to the default page size
  // - no matching properly aligned size class is available
  //
  // Requires that align is a non-zero power of 2.
  //
  // Specifying align = 1 will result in this method using the default
  // alignment of the size table. Calling this method with a constexpr
  // value of align = 1 will be optimized by the compiler, and result in
  // the inlined code to be identical to calling `GetSizeClass(size, cl)`
  inline bool ABSL_ATTRIBUTE_ALWAYS_INLINE GetSizeClass(size_t size,
                                                        size_t align,
                                                        uint32_t* cl) {
    ASSERT(align > 0);
    ASSERT((align & (align - 1)) == 0);

    if (ABSL_PREDICT_FALSE(align >= kPageSize)) {
      return false;
    }
    if (ABSL_PREDICT_FALSE(!GetSizeClass(size, cl))) {
      return false;
    }

    // Predict that size aligned allocs most often directly map to a proper
    // size class, i.e., multiples of 32, 64, etc, matching our class sizes.
    const size_t mask = (align - 1);
    do {
      if (ABSL_PREDICT_TRUE((class_to_size(*cl) & mask) == 0)) {
        return true;
      }
    } while (++*cl < kNumClasses);

    return false;
  }

  // Returns size class for given size, or 0 if this instance has not been
  // initialized yet. REQUIRES: size <= kMaxSize.
  inline size_t ABSL_ATTRIBUTE_ALWAYS_INLINE SizeClass(size_t size) {
    ASSERT(size <= kMaxSize);
    uint32_t ret = 0;
    GetSizeClass(size, &ret);
    return ret;
  }

  // Get the byte-size for a specified class. REQUIRES: cl <= kNumClasses.
  inline size_t ABSL_ATTRIBUTE_ALWAYS_INLINE class_to_size(size_t cl) {
    ASSERT(cl < kNumClasses);
    return class_to_size_[cl];
  }

  // Mapping from size class to number of pages to allocate at a time
  inline size_t class_to_pages(size_t cl) {
    ASSERT(cl < kNumClasses);
    return class_to_pages_[cl];
  }

  // Number of objects to move between a per-thread list and a central
  // list in one shot.  We want this to be not too small so we can
  // amortize the lock overhead for accessing the central list.  Making
  // it too big may temporarily cause unnecessary memory wastage in the
  // per-thread free list until the scavenger cleans up the list.
  inline SizeMap::BatchSize num_objects_to_move(size_t cl) {
    ASSERT(cl < kNumClasses);
    return num_objects_to_move_[cl];
  }
};

// Linker initialized, so this lock can be accessed at any time.
extern absl::base_internal::SpinLock pageheap_lock;

}  // namespace tcmalloc

#endif  // TCMALLOC_COMMON_H_
