#include "tensor.hpp"

#include <cstring>
#include <new>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <cstdlib>

// Storage class

Storage::Storage(std::size_t n)
    : size_(n) {

        // Validation
        if (n == 0){
            throw std::invalid_argument("Error: size must be greater than 0.");
        }

        if (n > SIZE_MAX / sizeof(float)){
            throw std::invalid_argument("Error: size parameter overflows size_t");
        }

        // Memory Allocation
        data_ = std::unique_ptr<float[], AlignedDeleter>(static_cast<float*>(::operator new(size_ * sizeof(float), std::align_val_t{ALIGNMENT_REQ}))); // Assign a temporary unique pointer over the raw pointer that operator new created

        // Zero out memory
        std::memset(data_.get(), 0, size_ * sizeof(float));
}

float *Storage::data() noexcept {
    return data_.get();
}

const float *Storage::data() const noexcept {
    return data_.get();
}

std::size_t Storage::size() const noexcept {
    return size_;
}

// Tensor class


std::array<std::size_t, MAX_RANK>
Tensor::checked_shape(std::initializer_list<std::size_t> ilist) {
    if (ilist.size() < 1 || ilist.size() > MAX_RANK) {
        throw std::invalid_argument("Error: Rank should be in [1, MAX_RANK].");
    }

    std::array<std::size_t, MAX_RANK> result{};

    std::size_t i = 0;
    for (std::size_t val : ilist) {
        if (val == 0) {
            throw std::invalid_argument("Error: every extent must be non-zero.");
        }
        result[i] = val;
        i++;
    }

    return result;
}

//Constructor 1
Tensor::Tensor(const std::array<std::size_t, MAX_RANK> &shape, std::size_t rank) {
    assert(rank >= 1 && rank <= MAX_RANK);
    for (std::size_t d = 0; d < rank; ++d) {
        assert(shape[d] != 0);
    }

    rank_ = rank;
    shape_ = shape;

    // Calculate numel (all the dimensions multiplied together)
    std::size_t numel = 1;

    for (std::size_t d = 0; d < rank_; d++){
        if (numel > SIZE_MAX / shape_[d]) {
            throw std::invalid_argument("Error: shape overflows when computing numel.");
        }
        numel *= shape_[d];
    }

    // Allocate storage_
    storage_ = std::make_shared<Storage>(numel);

    // Calculate the strides
    strides_[rank_ - 1] = 1;

    for (std::size_t d = rank_ - 1; d > 0; --d) {
        strides_[d - 1] = strides_[d] * shape_[d];
    }

    assert_within_storage();
}

// Constructor 2
Tensor::Tensor(std::initializer_list<std::size_t> ilist)
    : Tensor(checked_shape(ilist), ilist.size()) {}

// Constructor 3
Tensor::Tensor(std::shared_ptr<Storage> p, std::size_t offset,
               const std::array<std::size_t, MAX_RANK> & shape,
               const std::array<std::size_t, MAX_RANK> & strides, std::size_t rank) {
    
    storage_ = std::move(p);
    offset_ = offset;
    shape_ = shape;
    strides_ = strides;
    rank_ = rank;
    
    assert_within_storage();
}

/* ------------------------------------------------------------------------ */
/* Tensor - accessors                                                        */
/* ------------------------------------------------------------------------ */

std::size_t Tensor::rank() const noexcept {
    return rank_;
}

std::size_t Tensor::shape(std::size_t dim) const noexcept {
    assert(dim < rank_);
    return shape_[dim];
}

std::size_t Tensor::stride(std::size_t dim) const noexcept {
    assert(dim < rank_);
    return strides_[dim];
}

std::size_t Tensor::numel() const noexcept {
    std::size_t n = 1;
    for (std::size_t d = 0; d < rank_; ++d) {
        n *= shape_[d];
    }
    return n;
}

float *Tensor::data() noexcept {
    if (!storage_) {
        return nullptr;
    }
    return storage_->data() + offset_;
}

const float *Tensor::data() const noexcept {
    if (!storage_) {
        return nullptr;
    }
    return storage_->data() + offset_;
}

bool Tensor::is_contiguous() const noexcept {
    std::size_t expected = 1;
    for (std::size_t d = rank_; d > 0; --d) {
        std::size_t dim = d - 1;
        if (shape_[dim] != 1 && strides_[dim] != expected) {
            return false;
        }
        expected *= shape_[dim];
    }
    return true;
}

bool Tensor::shares_storage_with(const Tensor &other) const noexcept {
    return storage_ == other.storage_;
}

/* ------------------------------------------------------------------------ */
/* Tensor - whole-tensor operations                                          */
/* ------------------------------------------------------------------------ */

Tensor Tensor::clone() const {
    Tensor result(shape_, rank_);

    std::size_t n = numel();
    for (std::size_t lin = 0; lin < n; ++lin) {
        std::size_t remaining = lin;
        std::size_t src_offset = 0;

        for (std::size_t d = rank_; d > 0; --d) {
            std::size_t dim = d - 1;
            std::size_t idx = remaining % shape_[dim];
            remaining /= shape_[dim];
            src_offset += idx * strides_[dim];
        }

        result.data()[lin] = data()[src_offset];
    }

    return result;
}

void Tensor::zero() {
    if (!is_contiguous()){
        throw std::invalid_argument("Error: Should be contiguous");
    }

    std::memset(data(), 0 , numel() * sizeof(float));
}

void Tensor::randomize() {
    if (!is_contiguous()) {
        throw std::invalid_argument("Error: Should be contiguous");
    }

    std::size_t n = numel();
    float *ptr = data();

    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void Tensor::assert_within_storage() const noexcept {
    std::size_t max_offset = offset_;
    for (std::size_t d = 0; d < rank_; ++d) {
        max_offset += (shape_[d] - 1) * strides_[d];
    }
    assert(max_offset < storage_->size());
}
