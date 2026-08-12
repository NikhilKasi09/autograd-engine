#include "tensor_ops.hpp"

#include <array>
#include <stdexcept>
#include <string>


namespace {

// A tensor's shape and strides padded out to MAX_RANK for walking.
//
// Dimensions at or above rank get extent 1 and stride 0, so the four nested
// loops below run unconditionally: a rank-2 tensor trips the two outer loops
// exactly once each and contributes nothing to the offset. That is the
// alternative to decomposing a linear index into a multi-index per element,
// which costs a divide per dimension per element - clone() does it that way
// because it runs once, and these run every training step.
struct Walk {
    std::array<std::size_t, MAX_RANK> shape;
    std::array<std::size_t, MAX_RANK> strides;
};


Walk padded(const Tensor &t) {
    Walk w;
    w.shape.fill(1);
    w.strides.fill(0);

    for (std::size_t d = 0; d < t.rank(); ++d) {
        w.shape[d] = t.shape(d);
        w.strides[d] = t.stride(d);
    }

    return w;
}

// Throws unless a and b have identical rank and extents. `who` names the
// calling op so the message says which one rejected the shapes.
void require_same_shape(const Tensor &a, const Tensor &b, const char *who) {
    if (a.rank() != b.rank()) {
        throw std::invalid_argument(std::string(who) + ": rank mismatch");
    }
    for (std::size_t d = 0; d < a.rank(); ++d) {
        if (a.shape(d) != b.shape(d)) {
            throw std::invalid_argument(std::string(who) + ": shape mismatch");
        }
    }
}

// Throws unless out is contiguous. Writing through a non-contiguous
// destination is never meaningful here: an expand'd tensor has several logical
// elements aliasing one float, so the result would depend on iteration order.
void require_contiguous_out(const Tensor &out, const char *who) {
    if (!out.is_contiguous()) {
        throw std::invalid_argument(std::string(who) + ": output must be contiguous");
    }
}

} // namespace

/* ------------------------------------------------------------------------ */
/* Binary ops                                                                */
/* ------------------------------------------------------------------------ */

namespace {

template <typename BinaryOp>
void binary_elementwise(const Tensor &a, const Tensor &b, Tensor &out,
                          const char *who, BinaryOp op) {
    require_same_shape(a, b, who);
    require_same_shape(a, out, who);
    require_contiguous_out(out, who);

    Walk wa = padded(a);
    Walk wb = padded(b);
    Walk wo = padded(out);

    float *out_ptr = out.data();
    const float *a_ptr = a.data();
    const float *b_ptr = b.data();

    std::size_t lin = 0;
    for (std::size_t i0 = 0; i0 < wo.shape[0]; i0++) {
        for (std::size_t i1 = 0; i1 < wo.shape[1]; i1++) {
            for (std::size_t i2 = 0; i2 < wo.shape[2]; i2++) {
                for (std::size_t i3 = 0; i3 < wo.shape[3]; i3++) {
                    std::size_t a_off = i0*wa.strides[0] + i1*wa.strides[1] + i2*wa.strides[2] + i3*wa.strides[3];
                    std::size_t b_off = i0*wb.strides[0] + i1*wb.strides[1] + i2*wb.strides[2] + i3*wb.strides[3];

                    out_ptr[lin] = op(a_ptr[a_off], b_ptr[b_off]);
                    lin++;
                }
            }
        }
    }
}

} // namespace

void add(const Tensor &a, const Tensor &b, Tensor &out) {
    binary_elementwise(a, b, out, "add", [](float x, float y) { return x + y; });
}

void mul(const Tensor &a, const Tensor &b, Tensor &out) {
    binary_elementwise(a, b, out, "mul", [](float x, float y) { return x * y; });
}

/* ------------------------------------------------------------------------ */
/* Unary ops                                                                 */
/* ------------------------------------------------------------------------ */

namespace{

    template <typename UnaryOp>
    void unary_elementwise(const Tensor &a, Tensor &out, const char *who, UnaryOp op){

        require_same_shape(a, out, who);
        require_contiguous_out(out, who);

        Walk wa = padded(a);
        Walk wo = padded(out);

        float *out_ptr = out.data();
        const float *a_ptr = a.data();

        std::size_t lin = 0;
        for (std::size_t i0 = 0; i0 < wo.shape[0]; i0++) {
            for (std::size_t i1 = 0; i1 < wo.shape[1]; i1++) {
                for (std::size_t i2 = 0; i2 < wo.shape[2]; i2++) {
                    for (std::size_t i3 = 0; i3 < wo.shape[3]; i3++) {
                        std::size_t a_off = i0*wa.strides[0] + i1*wa.strides[1] + i2*wa.strides[2] + i3*wa.strides[3];

                        out_ptr[lin] = op(a_ptr[a_off]);
                        lin++;
                    }
                }
            }
        }
    }
}

void scale(const Tensor &a, float s, Tensor &out) {
    unary_elementwise(a, out, "scale", [s](float x) { return x * s; });
}

void relu(const Tensor &a, Tensor &out) {
    unary_elementwise(a, out, "relu", [](float x) { return x > 0.0f ? x : 0.0f; });
}

/* ------------------------------------------------------------------------ */
/* Accumulate and reduce                                                     */
/* ------------------------------------------------------------------------ */

void add_into(Tensor &dst, const Tensor &src) {
    require_same_shape(dst, src, "add_into");
    require_contiguous_out(dst, "add_into");

    Walk ws = padded(src);

    float *dst_ptr = dst.data();
    const float *src_ptr = src.data();

    std::size_t lin = 0;
    for (std::size_t i0 = 0; i0 < ws.shape[0]; i0++) {
        for (std::size_t i1 = 0; i1 < ws.shape[1]; i1++) {
            for (std::size_t i2 = 0; i2 < ws.shape[2]; i2++) {
                for (std::size_t i3 = 0; i3 < ws.shape[3]; i3++) {
                    std::size_t src_off = i0*ws.strides[0] + i1*ws.strides[1] + i2*ws.strides[2] + i3*ws.strides[3];

                    dst_ptr[lin] += src_ptr[src_off];
                    lin++;
                }
            }
        }
    }
}

float sum(const Tensor &src) {// Collapses the entire tensor down to a single number
    Walk ws = padded(src);
    const float *src_ptr = src.data();

    double total = 0.0;
    for (std::size_t i0 = 0; i0 < ws.shape[0]; i0++) {
        for (std::size_t i1 = 0; i1 < ws.shape[1]; i1++) {
            for (std::size_t i2 = 0; i2 < ws.shape[2]; i2++) {
                for (std::size_t i3 = 0; i3 < ws.shape[3]; i3++) {
                    std::size_t src_off = i0*ws.strides[0] + i1*ws.strides[1] + i2*ws.strides[2] + i3*ws.strides[3];

                    total += src_ptr[src_off];
                }
            }
        }
    }

    return static_cast<float>(total);
}
