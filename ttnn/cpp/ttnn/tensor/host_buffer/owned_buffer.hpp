// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

namespace tt::tt_metal::owned_buffer {

template <typename T>
struct Buffer {
    using value_type = T;

    explicit Buffer() = default;
    explicit Buffer(std::shared_ptr<std::vector<T>>&& shared_vector) :
        shared_vector_(shared_vector),
        pointer_for_faster_access_(shared_vector->data()),
        size_(shared_vector->size()) {}
    explicit Buffer(const std::shared_ptr<std::vector<T>>& shared_vector) :
        shared_vector_(shared_vector),
        pointer_for_faster_access_(shared_vector->data()),
        size_(shared_vector->size()) {}

    const std::size_t size() const { return this->size_; }

    inline T& operator[](std::size_t index) noexcept { return this->pointer_for_faster_access_[index]; }
    inline const T& operator[](std::size_t index) const noexcept { return this->pointer_for_faster_access_[index]; }

    inline T* begin() noexcept { return this->pointer_for_faster_access_; }
    inline T* end() noexcept { return this->pointer_for_faster_access_ + this->size(); }

    inline const T* begin() const noexcept { return this->pointer_for_faster_access_; }
    inline const T* end() const noexcept { return this->pointer_for_faster_access_ + this->size(); }

    inline bool is_allocated() const { return bool(this->shared_vector_); }
    inline const std::vector<T>& get() const { return *this->shared_vector_; }
    inline const std::shared_ptr<std::vector<T>> get_ptr() const noexcept { return this->shared_vector_; }
    inline void reset() { this->shared_vector_.reset(); }

    inline void* data() noexcept { return static_cast<void*>(this->pointer_for_faster_access_); }
    inline const void* data() const noexcept { return static_cast<void*>(this->pointer_for_faster_access_); }
    inline uint32_t use_count() const noexcept { return this->shared_vector_.use_count(); }

private:
    std::shared_ptr<std::vector<T>> shared_vector_;
    T* pointer_for_faster_access_ = nullptr;
    std::size_t size_ = 0;
};

template <typename T>
bool operator==(const Buffer<T>& buffer_a, const Buffer<T>& buffer_b) noexcept {
    if (buffer_a.size() != buffer_b.size()) {
        return false;
    }
    for (auto index = 0; index < buffer_a.size(); index++) {
        if (buffer_a[index] != buffer_b[index]) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool operator!=(const Buffer<T>& buffer_a, const Buffer<T>& buffer_b) noexcept {
    return not(buffer_a == buffer_b);
}

}  // namespace tt::tt_metal::owned_buffer
