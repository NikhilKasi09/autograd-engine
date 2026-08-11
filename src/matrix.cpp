#include "matrix.hpp"
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX 


#include <cstring>
#include <new>
#include <stdexcept>


// Matrix - RAII replacement for matrix_t.
// AlignedDeleter's body moved to aligned.hpp, where Tensor can share it.


Matrix::Matrix(std::size_t rows, std::size_t cols)
    :  rows_(rows), cols_(cols), stride_(cols)
    {

    // Validation
    if (rows == 0 || cols == 0){
        throw std::invalid_argument("Error: Matrix dimensions must be greater than 0.\n");
    }

    if (rows > SIZE_MAX / cols / sizeof(float)){
        throw std::invalid_argument("Error: Matrix dimensions overflow size_t");
    }

    // Memory Allocation
    std::size_t num_bytes = rows * cols * sizeof(float);
    data_ = std::unique_ptr<float[], AlignedDeleter>(static_cast<float*>(::operator new(num_bytes, std::align_val_t{ALIGNMENT_REQ}))); // Assign a temporary unique pointer over the raw pointer that operator new created

    // Zero out memory
    std::memset(data_.get(), 0, num_bytes);
}

Matrix Matrix::clone() const {
    
    Matrix other(rows_, cols_);

    for(std::size_t i = 0; i < rows_ ; i++){
            std::memcpy(other.data() + i * other.stride(), data_.get() + i * stride_, cols_ * sizeof(float));
    }

    return other;
}

float *Matrix::data() noexcept {
    return data_.get();
}

const float *Matrix::data() const noexcept { // used if you want to look at data from a read only matrix
    return data_.get();
}

std::size_t Matrix::rows() const noexcept {
    return rows_;
}

std::size_t Matrix::cols() const noexcept {
    return cols_;
}

std::size_t Matrix::stride() const noexcept {
    return stride_;
}

void Matrix::zero() {
    for (std::size_t i = 0; i < rows_; i++) {
        std::memset(data_.get() + i * stride_, 0, cols_ * sizeof(float));
    }
}

void Matrix::randomize() {
    for (std::size_t i = 0; i < rows_; i++){
        for (std::size_t j = 0; j < cols_; j++){
            data_.get()[i * stride_ + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}
