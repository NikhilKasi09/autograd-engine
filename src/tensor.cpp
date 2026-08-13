#include "tensor.hpp"

#include <cstring>
#include <new>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <utility>

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

//Private helpers
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

std::array<std::size_t, MAX_RANK>
Tensor::checked_shape(std::span<const std::size_t> shape) {
    if (shape.size() < 1 || shape.size() > MAX_RANK) {
        throw std::invalid_argument("Error: Rank should be in [1, MAX_RANK].");
    }

    std::array<std::size_t, MAX_RANK> result{};

    std::size_t i = 0;
    for (std::size_t val : shape) {
        if (val == 0) {
            throw std::invalid_argument("Error: every extent must be non-zero.");
        }
        result[i] = val;
        i++;
    }

    return result;
}

std::size_t Tensor::numel_of(const std::array<std::size_t, MAX_RANK> &shape, std::size_t rank) {
    std::size_t n = 1;
    for (std::size_t d = 0; d < rank; ++d) {
        if (n > SIZE_MAX / shape[d]) {
            throw std::invalid_argument("Error: shape overflows when computing numel.");
        }
        n *= shape[d];
    }
    return n;
}

std::array<std::size_t, MAX_RANK>
Tensor::row_major_strides(const std::array<std::size_t, MAX_RANK> &shape, std::size_t rank) {
    std::array<std::size_t, MAX_RANK> strides{};
    strides[rank - 1] = 1;
    for (std::size_t d = rank - 1; d > 0; --d) {
        strides[d - 1] = strides[d] * shape[d];
    }
    return strides;
}

//Constructor 1
Tensor::Tensor(const std::array<std::size_t, MAX_RANK> &shape, std::size_t rank) {
    assert(rank >= 1 && rank <= MAX_RANK);
    for (std::size_t d = 0; d < rank; ++d) {
        assert(shape[d] != 0);
    }

    rank_ = rank;
    shape_ = shape;

    std::size_t numel = numel_of(shape_, rank_);
    storage_ = std::make_shared<Storage>(numel);
    strides_ = row_major_strides(shape_, rank_);

    assert_within_storage();
}

// Constructor 2
Tensor::Tensor(std::initializer_list<std::size_t> ilist)
    : Tensor(std::span<const std::size_t>(ilist.begin(), ilist.size())) {}

// Constructor 2b - the runtime-shape door.
Tensor::Tensor(std::span<const std::size_t> shape)
    : Tensor(checked_shape(shape), shape.size()) {}

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

/* ------------------------------------------------------------------------ */
/* Tensor - views                                                            */
/*                                                                           */
/* Every one of these ends the same way: build a local shape and strides,     */
/* then return through the view constructor. None of them touches shape_,     */
/* strides_ or offset_ on a copy of *this - going through the constructor is  */
/* what makes assert_within_storage() impossible to skip.                     */
/* ------------------------------------------------------------------------ */

Tensor Tensor::transpose(std::size_t d0, std::size_t d1) const { // transposes the d0th and d1st dimensions 

    if (d0 >= rank_ || d1 >= rank_){
        throw std::invalid_argument("Error: dimensions to be tranposed are 0 indexed and must be lower than the rank.");
    }

    std::array<std::size_t, MAX_RANK> this_shape = shape_;
    std::array<std::size_t, MAX_RANK> this_strides = strides_;

    std::swap(this_shape[d0], this_shape[d1]);
    std::swap(this_strides[d0], this_strides[d1]);

    return Tensor(storage_, offset_, this_shape, this_strides, rank_);
}

Tensor Tensor::permute(std::initializer_list<std::size_t> dims) const {
    return permute(std::span<const std::size_t>(dims.begin(), dims.size()));
}

Tensor Tensor::permute(std::span<const std::size_t> dims) const {
    if (dims.size() != rank_) {
        throw std::invalid_argument("Error: permute requires the same rank.");
    }

    std::array<bool, MAX_RANK> seen{};  // all false by default

    std::array<std::size_t, MAX_RANK> new_shape{};
    std::array<std::size_t, MAX_RANK> new_strides{};

    std::size_t i = 0;
    for (std::size_t d : dims) {
        if (d >= rank_ || seen[d]) {
            throw std::invalid_argument("Error: dims must be a permutation of 0..rank_-1.");
        }
        seen[d] = true;

        new_shape[i] = shape_[d];
        new_strides[i] = strides_[d];
        i++;
    }

    return Tensor(storage_, offset_, new_shape, new_strides, rank_);
}

Tensor Tensor::slice(std::size_t dim, std::size_t start, std::size_t count) const { // Picks a sub-range our of one dimension, leaving the rest of the tensor untouched

    if (dim >= rank_){
        throw std::invalid_argument("Error: dimension to be sliced is 0 indexed and should be lower than the rank.");
    }

    if (count == 0){
        throw std::invalid_argument("Error: Zero-length slice is invalid.");
    }

    if (start >= shape_[dim]) {
        throw std::invalid_argument("Error: start is out of range.");
    }
    
    if (count > shape_[dim] - start) {
        throw std::invalid_argument("Error: slice runs off the end of the dimension.");
    }

    std::size_t new_offset = offset_ + start * strides_[dim]; // Remember its pointing to the same actual memory so offset needs to be updated
    std::array<std::size_t, MAX_RANK> this_shape = shape_;
    this_shape[dim] = count; // sliced dimension shrinks to requested count 

    return Tensor(storage_, new_offset, this_shape, strides_, rank_);
}

Tensor Tensor::expand(std::initializer_list<std::size_t> new_shape) const {
    return expand(std::span<const std::size_t>(new_shape.begin(), new_shape.size())); 
}

Tensor Tensor::expand(std::span<const std::size_t> new_shape) const {
    if (new_shape.size() != rank_) {
        throw std::invalid_argument("Error: expand requires the same rank.");
    }

    std::array<std::size_t, MAX_RANK> this_shape = shape_;
    std::array<std::size_t, MAX_RANK> this_strides = strides_;

    std::size_t d = 0;
    for (std::size_t extent : new_shape) {
        if (extent == shape_[d]) {
            // unchanged: keep current extent and stride
        } else if (shape_[d] == 1) {
            if (extent == 0) {
                throw std::invalid_argument("Error: expand cannot produce a zero extent.");
            }
            this_shape[d] = extent;
            this_strides[d] = 0;
        } else {
            throw std::invalid_argument("Error: cannot expand a dimension with extent > 1.");
        }
        d++;
    }

    return Tensor(storage_, offset_, this_shape, this_strides, rank_);
}

Tensor Tensor::reshape(std::initializer_list<std::size_t> new_shape) const { // Reinterprets a tensors data under a completely different shape
    
    return reshape(std::span<const std::size_t>(new_shape.begin(), new_shape.size()));
}

Tensor Tensor::reshape(std::span<const std::size_t> new_shape) const {
    if (!is_contiguous()){
        throw std::invalid_argument("Error: Tensor has to be contiguous");
    }

    std::array<std::size_t, MAX_RANK> shape = checked_shape(new_shape);
    std::size_t rank = new_shape.size();

    if (numel_of(shape, rank) != numel_of(shape_, rank_)){
        throw std::invalid_argument("Error: Number of elements should be the same when reshaping.");
    }

    std::array<std::size_t, MAX_RANK>strides = row_major_strides(shape, rank);

    return Tensor(storage_, offset_, shape, strides, rank);
}

Tensor Tensor::contiguous() const { // returns by value - creates a copy
    
    if (is_contiguous()){
        return *this; // dereference and return this, which creates a copy
    } else {
        return clone();
    }
}

void Tensor::assert_within_storage() const noexcept {
    std::size_t max_offset = offset_;
    for (std::size_t d = 0; d < rank_; ++d) {
        max_offset += (shape_[d] - 1) * strides_[d];
    }
    assert(max_offset < storage_->size());
}
