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
// A data structure used by the caching malloc.  It maps from page# to
// a pointer that contains info about that page using a two-level array.
//
// The BITS parameter should be the number of bits required to hold
// a page number.  E.g., with 32 bit pointers and 8K pages (i.e.,
// page offset fits in lower 13 bits), BITS == 19.
//
// A PageMap requires external synchronization, except for the get/sizeclass
// methods (see explanation at top of tcmalloc.cc).

#ifndef TCMALLOC_PAGEMAP_H_
#define TCMALLOC_PAGEMAP_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/logging.h"
#include "tcmalloc/span.h"
#include "tcmalloc/static_vars.h"

namespace tcmalloc {

// Two-level radix tree
typedef void* (*PagemapAllocator)(size_t);
void* MetaDataAlloc(size_t bytes);


/// <summary>
/// PageMap 主要用于维护 page -> span 的映射
/// </summary>
/// <typeparam name="BITS"></typeparam>
/// <typeparam name="Allocator"></typeparam>
template <int BITS, PagemapAllocator Allocator>
class PageMap2 {
 private:
  // The leaf node (regardless of pointer size) always maps 2^15 entries;
  // with 8K pages, this gives us 256MB mapped per leaf node.
  // 246M / 8K = 32K = 2^15, 一个leaf有32k个page
  static const int kLeafBits = 15; 
  static const int kLeafLength = 1 << kLeafBits;
  // BITS = all virtual addr - page shift
  // BITS - kLeafBits = all Leaf number
  static const int kRootBits = (BITS >= kLeafBits) ? (BITS - kLeafBits) : 0;
  // (1<<kRootBits) must not overflow an "int"
  static_assert(kRootBits < sizeof(int) * 8 - 1, "kRootBits is too large");
  static const int kRootLength = 1 << kRootBits;

  static const size_t kLeafCoveredBytes = 1ul << (kLeafBits + kPageShift);
  static_assert(kLeafCoveredBytes >= kHugePageSize, "leaf too small");
  static const size_t kLeafHugeBits = (kLeafBits + kPageShift - kHugePageShift);
  static const size_t kLeafHugepages = kLeafCoveredBytes / kHugePageSize;
  static_assert(kLeafHugepages == 1 << kLeafHugeBits, "sanity");
  struct Leaf {
    // We keep parallel arrays indexed by page number.  One keeps the
    // size class; another span pointers; the last hugepage-related
    // information.  The size class information is kept segregated
    // since small object deallocations are so frequent and do not
    // need the other information kept in a Span.
    uint8_t sizeclass[kLeafLength];
    void* span[kLeafLength];
    void* hugepage[kLeafHugepages];
  };

  // 将整个虚拟地址,分割为 page / Leaf
  // kRootLength为整个虚拟地址所能分割的Leaf的个数,即全地址空间映射
  // 再分一层Leaf的好处是可以让初始化申请的数组长度大幅度减少,每个Leaf涵盖的
  // 地址范围所需要的span指针的申请可以延迟到这个Leaf范围有span加入时
  Leaf* root_[kRootLength];             // Top-level node
  size_t bytes_used_;

 public:
  typedef uintptr_t Number;

  constexpr PageMap2() : root_{}, bytes_used_(0) {}

  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  void* get(Number k) const NO_THREAD_SAFETY_ANALYSIS {
  // 通过PageID找到span
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    if ((k >> BITS) > 0 || root_[i1] == nullptr) {
      return nullptr;
    }
    return root_[i1]->span[i2];
  }

  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  // Requires that the span is known to already exist.
  void* get_existing(Number k) const NO_THREAD_SAFETY_ANALYSIS {
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    ASSERT((k >> BITS) == 0);
    ASSERT(root_[i1] != nullptr);
    return root_[i1]->span[i2];
  }

  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  // REQUIRES: Must be a valid page number previously Ensure()d.
  uint8_t ABSL_ATTRIBUTE_ALWAYS_INLINE
  sizeclass(Number k) const NO_THREAD_SAFETY_ANALYSIS {
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    ASSERT((k >> BITS) == 0);
    ASSERT(root_[i1] != nullptr);
    return root_[i1]->sizeclass[i2];
  }

  void set(Number k, void* v) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    root_[i1]->span[i2] = v;
  }

  void set_with_sizeclass(Number k, void* v, uint8_t sc) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    Leaf* leaf = root_[i1];
    leaf->span[i2] = v;
    leaf->sizeclass[i2] = sc;
  }

  void clear_sizeclass(Number k) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    root_[i1]->sizeclass[i2] = 0;
  }

  void* get_hugepage(Number k) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    if ((k >> BITS) > 0 || root_[i1] == nullptr) {
      return nullptr;
    }
    return root_[i1]->hugepage[i2 >> (kLeafBits - kLeafHugeBits)];
  }

  void set_hugepage(Number k, void* v) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> kLeafBits;
    const Number i2 = k & (kLeafLength - 1);
    root_[i1]->hugepage[i2 >> (kLeafBits - kLeafHugeBits)] = v;
  }

  bool Ensure(Number start, size_t n) {
    ASSERT(n > 0);
    // 这个函数用于确保申请的这片内存用于维护的Leaf的存在
    for (Number key = start; key <= start + n - 1; ) {
      const Number i1 = key >> kLeafBits;

      // Check for overflow
      if (i1 >= kRootLength) return false;

      // Make 2nd level node if necessary
      if (root_[i1] == nullptr) {
        Leaf* leaf = reinterpret_cast<Leaf*>(Allocator(sizeof(Leaf)));
        if (leaf == nullptr) return false;
        bytes_used_ += sizeof(Leaf);
        memset(leaf, 0, sizeof(*leaf));
        root_[i1] = leaf;
      }

      // Advance key past whatever is covered by this leaf node
      key = ((key >> kLeafBits) + 1) << kLeafBits;
    }
    return true;
  }

  size_t bytes_used() const {
    // Account for size of root node, etc.
    return bytes_used_ + sizeof(*this);
  }

  constexpr size_t RootSize() const { return sizeof(root_); }
  const void* RootAddress() { return root_; }
};

// Three-level radix tree
// Currently only used for TCMALLOC_SMALL_BUT_SLOW
template <int BITS, PagemapAllocator Allocator>
class PageMap3 {
 private:
  // For x86 we currently have 48 usable bits, for POWER we have 46. With
  // 4KiB page sizes (12 bits) we end up with 36 bits for x86 and 34 bits
  // for POWER. So leaf covers 4KiB * 1 << 12 = 16MiB - which is huge page
  // size for POWER.
  static const int kLeafBits = (BITS + 2) / 3;  // Round up
  static const int kLeafLength = 1 << kLeafBits;
  static const int kMidBits = (BITS + 2) / 3;  // Round up
  static const int kMidLength = 1 << kMidBits;
  static const int kRootBits = BITS - kLeafBits - kMidBits;
  static_assert(kRootBits > 0, "Too many bits assigned to leaf and mid");
  // (1<<kRootBits) must not overflow an "int"
  static_assert(kRootBits < sizeof(int) * 8 - 1, "Root bits too large");
  static const int kRootLength = 1 << kRootBits;

  static const size_t kLeafCoveredBytes = size_t{1} << (kLeafBits + kPageShift);
  static_assert(kLeafCoveredBytes >= kHugePageSize, "leaf too small");
  static const size_t kLeafHugeBits = (kLeafBits + kPageShift - kHugePageShift);
  static const size_t kLeafHugepages = kLeafCoveredBytes / kHugePageSize;
  static_assert(kLeafHugepages == 1 << kLeafHugeBits, "sanity");
  struct Leaf {
    // We keep parallel arrays indexed by page number.  One keeps the
    // size class; another span pointers; the last hugepage-related
    // information.  The size class information is kept segregated
    // since small object deallocations are so frequent and do not
    // need the other information kept in a Span.
    uint8_t sizeclass[kLeafLength];
    void* span[kLeafLength];
    void* hugepage[kLeafHugepages];
  };

  struct Node {
    // Mid-level structure that holds pointers to leafs
    Leaf* leafs[kMidLength];
  };

  Node* root_[kRootLength];  // Top-level node
  size_t bytes_used_;

 public:
  typedef uintptr_t Number;

  constexpr PageMap3() : root_{}, bytes_used_(0) {}

  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  void* get(Number k) const NO_THREAD_SAFETY_ANALYSIS {
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    if ((k >> BITS) > 0 || root_[i1] == nullptr ||
        root_[i1]->leafs[i2] == nullptr) {
      return nullptr;
    }
    return root_[i1]->leafs[i2]->span[i3];
  }

  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  // Requires that the span is known to already exist.
  void* get_existing(Number k) const NO_THREAD_SAFETY_ANALYSIS {
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    ASSERT((k >> BITS) == 0);
    ASSERT(root_[i1] != nullptr);
    ASSERT(root_[i1]->leafs[i2] != nullptr);
    return root_[i1]->leafs[i2]->span[i3];
  }

  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  // REQUIRES: Must be a valid page number previously Ensure()d.
  uint8_t ABSL_ATTRIBUTE_ALWAYS_INLINE
  sizeclass(Number k) const NO_THREAD_SAFETY_ANALYSIS {
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    ASSERT((k >> BITS) == 0);
    ASSERT(root_[i1] != nullptr);
    ASSERT(root_[i1]->leafs[i2] != nullptr);
    return root_[i1]->leafs[i2]->sizeclass[i3];
  }

  void set(Number k, void* v) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    root_[i1]->leafs[i2]->span[i3] = v;
  }

  void set_with_sizeclass(Number k, void* v, uint8_t sc) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    Leaf* leaf = root_[i1]->leafs[i2];
    leaf->span[i3] = v;
    leaf->sizeclass[i3] = sc;
  }

  void clear_sizeclass(Number k) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    root_[i1]->leafs[i2]->sizeclass[i3] = 0;
  }

  void* get_hugepage(Number k) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    if ((k >> BITS) > 0 || root_[i1] == nullptr ||
        root_[i1]->leafs[i2] == nullptr) {
      return nullptr;
    }
    return root_[i1]->leafs[i2]->hugepage[i3 >> (kLeafBits - kLeafHugeBits)];
  }

  void set_hugepage(Number k, void* v) {
    ASSERT(k >> BITS == 0);
    const Number i1 = k >> (kLeafBits + kMidBits);
    const Number i2 = (k >> kLeafBits) & (kMidLength - 1);
    const Number i3 = k & (kLeafLength - 1);
    root_[i1]->leafs[i2]->hugepage[i3 >> (kLeafBits - kLeafHugeBits)] = v;
  }

  bool Ensure(Number start, size_t n) {
    for (Number key = start; key <= start + n - 1;) {
      const Number i1 = key >> (kLeafBits + kMidBits);
      const Number i2 = (key >> kLeafBits) & (kMidLength - 1);

      // Check within root
      if (i1 >= kRootLength) return false;

      // Allocate Node if necessary
      if (root_[i1] == nullptr) {
        Node* node = reinterpret_cast<Node*>(Allocator(sizeof(Node)));
        if (node == nullptr) return false;
        bytes_used_ += sizeof(Node);
        memset(node, 0, sizeof(*node));
        root_[i1] = node;
      }

      // Allocate Leaf if necessary
      if (root_[i1]->leafs[i2] == nullptr) {
        Leaf* leaf = reinterpret_cast<Leaf*>(Allocator(sizeof(Leaf)));
        if (leaf == nullptr) return false;
        bytes_used_ += sizeof(Leaf);
        memset(leaf, 0, sizeof(*leaf));
        root_[i1]->leafs[i2] = leaf;
      }

      // Advance key past whatever is covered by this leaf node
      key = ((key >> kLeafBits) + 1) << kLeafBits;
    }
    return true;
  }

  size_t bytes_used() const { return bytes_used_ + sizeof(*this); }

  constexpr size_t RootSize() const { return sizeof(root_); }
  const void* RootAddress() { return root_; }
};

class PageMap {
 public:
  constexpr PageMap() : map_{} {}

  // Return the size class for p, or 0 if it is not known to tcmalloc
  // or is a page containing large objects.
  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  uint8_t sizeclass(PageID p) NO_THREAD_SAFETY_ANALYSIS {
    return map_.sizeclass(p);
  }

  void Set(PageID p, Span* span) { map_.set(p, span); }

  bool Ensure(PageID p, size_t n) EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return map_.Ensure(p, n);
  }

  // Mark an allocated span as being used for small objects of the
  // specified size-class.
  // REQUIRES: span was returned by an earlier call to PageAllocator::New()
  //           and has not yet been deleted.
  // Concurrent calls to this method are safe unless they mark the same span.
  void RegisterSizeClass(Span* span, size_t sc);

  // Mark an allocated span as being not used for any size-class.
  // REQUIRES: span was returned by an earlier call to PageAllocator::New()
  //           and has not yet been deleted.
  // Concurrent calls to this method are safe unless they mark the same span.
  void UnregisterSizeClass(Span* span);

  // Return the descriptor for the specified page.  Returns NULL if
  // this PageID was not allocated previously.
  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  inline Span* GetDescriptor(PageID p) const NO_THREAD_SAFETY_ANALYSIS {
    return reinterpret_cast<Span*>(map_.get(p));
  }

  // Return the descriptor for the specified page.
  // PageID must have been previously allocated.
  // No locks required.  See SYNCHRONIZATION explanation at top of tcmalloc.cc.
  ABSL_ATTRIBUTE_RETURNS_NONNULL inline Span* GetExistingDescriptor(
      PageID p) const NO_THREAD_SAFETY_ANALYSIS {
    Span* span = reinterpret_cast<Span*>(map_.get_existing(p));
    ASSERT(span != nullptr);
    return span;
  }

  size_t bytes() const EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return map_.bytes_used();
  }

  void* GetHugepage(PageID p) { return map_.get_hugepage(p); }

  void SetHugepage(PageID p, void* v) { map_.set_hugepage(p, v); }

  // The PageMap root node can be quite large and sparsely used. If this
  // gets mapped with hugepages we potentially end up holding a large
  // amount of unused memory. So it is better to map the root node with
  // small pages to minimise the amount of unused memory.
  void MapRootWithSmallPages();

 private:
// radix tree
#ifdef TCMALLOC_USE_PAGEMAP3
  PageMap3<kAddressBits - kPageShift, MetaDataAlloc> map_;
#else
  PageMap2<kAddressBits - kPageShift, MetaDataAlloc> map_;
#endif
};

}  // namespace tcmalloc

#endif  // TCMALLOC_PAGEMAP_H_
