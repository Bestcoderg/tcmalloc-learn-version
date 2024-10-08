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

#ifndef TCMALLOC_PAGE_HEAP_ALLOCATOR_H_
#define TCMALLOC_PAGE_HEAP_ALLOCATOR_H_

#include <stddef.h>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "tcmalloc/arena.h"
#include "tcmalloc/common.h"
#include "tcmalloc/internal/logging.h"

namespace tcmalloc {

struct AllocatorStats {
  // Number of allocated but unfreed objects
  size_t in_use;
  // Number of objects created (both free and allocated)
  size_t total;
};

// Simple allocator for objects of a specified type.  External locking
// is required before accessing one of these objects.
template <class T>
class PageHeapAllocator {
 public:
  // We use an explicit Init function because these variables are statically
  // allocated and their constructors might not have run by the time some
  // other static variable tries to allocate memory.
  void Init(Arena* arena) EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    arena_ = arena;
    stats_ = {0, 0};
    free_list_ = nullptr;
    // Reserve some space at the beginning to avoid fragmentation.
    Delete(New());
  }

  T* New() EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    // Consult free list
    T* result = free_list_;
    stats_.in_use++;
    if (ABSL_PREDICT_FALSE(result == nullptr)) {
      stats_.total++;
      return reinterpret_cast<T*>(arena_->Alloc(sizeof(T)));
    }
    free_list_ = *(reinterpret_cast<T**>(free_list_));
    return result;
  }

  void Delete(T* p) EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    /// 将free_list直接存入这个空节点内存中，然后将free_list的头指针指向这个节点地址，相当于将空节点内存复用为free_list
    *(reinterpret_cast<void**>(p)) = free_list_;
    free_list_ = p;
    stats_.in_use--;
  }

  AllocatorStats stats() const EXCLUSIVE_LOCKS_REQUIRED(pageheap_lock) {
    return stats_;
  }

 private:
  // Arena from which to allocate memory
  // 对于每个模板类的内存申请再做优化,每次的内存申请走Arena,Arena仅增长不回收
  Arena* arena_;

  // Free list of already carved objects
  T* free_list_ GUARDED_BY(pageheap_lock);

  AllocatorStats stats_ GUARDED_BY(pageheap_lock);
};

}  // namespace tcmalloc

#endif  // TCMALLOC_PAGE_HEAP_ALLOCATOR_H_
