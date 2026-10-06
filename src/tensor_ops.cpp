#include "tensor_ops.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>


namespace {

// A tensor's shape and strides padded out to MAX_RANK for walking.
//
// The padding goes on the LEFT: a rank-2 tensor fills slots 2 and 3, and slots
// 0 and 1 get extent 1 and stride 0. So the four nested loops below run
// unconditionally, the two outer ones trip once each, and the tensor's last
// dimension is always the innermost loop - the one worth making fast.
//
// Padding on the right gives the same answers and put a rank-2 tensor's
// columns in loop 1, leaving the innermost loop a single trip per element.
struct Walk {
    std::array<std::size_t, MAX_RANK> shape;
    std::array<std::size_t, MAX_RANK> strides;
};


Walk padded(const Tensor &t) {
    Walk w;
    w.shape.fill(1);
    w.strides.fill(0);

    // Dimension d lands in slot d + pad, so the last one lands in slot 3.
    const std::size_t pad = MAX_RANK - t.rank();
    for (std::size_t d = 0; d < t.rank(); ++d) {
        w.shape[d + pad] = t.shape(d);
        w.strides[d + pad] = t.stride(d);
    }

    return w;
}

// Offset of the first element of the row at (i0, i1, i2). A row is one run
// along the last dimension; its elements are strides[3] apart from here.
inline std::size_t row_start(const Walk &w, std::size_t i0, std::size_t i1, std::size_t i2) {
    return i0*w.strides[0] + i1*w.strides[1] + i2*w.strides[2];
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

    const float *a_ptr = a.data();
    const float *b_ptr = b.data();

    // out is contiguous, so its rows follow one another: this just advances.
    float *out_row = out.data();

    const std::size_t n = wo.shape[3];      // elements in one row
    const std::size_t sa = wa.strides[3];   // step along a row, per input
    const std::size_t sb = wb.strides[3];

    for (std::size_t i0 = 0; i0 < wo.shape[0]; i0++) {
        for (std::size_t i1 = 0; i1 < wo.shape[1]; i1++) {
            for (std::size_t i2 = 0; i2 < wo.shape[2]; i2++) {
                const float *a_row = a_ptr + row_start(wa, i0, i1, i2);
                const float *b_row = b_ptr + row_start(wb, i0, i1, i2);

                if (sa == 1 && sb == 1) {
                    // Both rows are contiguous runs. With no stride in the
                    // index the compiler can vectorise this one.
                    for (std::size_t j = 0; j < n; j++) {
                        out_row[j] = op(a_row[j], b_row[j]);
                    }
                } else {
                    // A transposed input, or stride 0 from an expand.
                    for (std::size_t j = 0; j < n; j++) {
                        out_row[j] = op(a_row[j*sa], b_row[j*sb]);
                    }
                }

                out_row += n;
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

void relu_backward(const Tensor &grad_out, const Tensor &ref, Tensor &out) {
    binary_elementwise(grad_out, ref, out, "relu_backward",
                       [](float g, float r) { return r > 0.0f ? g : 0.0f; });
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

        const float *a_ptr = a.data();
        float *out_row = out.data();            // contiguous: rows are adjacent

        const std::size_t n = wo.shape[3];      // elements in one row
        const std::size_t sa = wa.strides[3];   // step along a row of a

        for (std::size_t i0 = 0; i0 < wo.shape[0]; i0++) {
            for (std::size_t i1 = 0; i1 < wo.shape[1]; i1++) {
                for (std::size_t i2 = 0; i2 < wo.shape[2]; i2++) {
                    const float *a_row = a_ptr + row_start(wa, i0, i1, i2);

                    if (sa == 1) {
                        // A contiguous run: the vectorisable case.
                        for (std::size_t j = 0; j < n; j++) {
                            out_row[j] = op(a_row[j]);
                        }
                    } else {
                        for (std::size_t j = 0; j < n; j++) {
                            out_row[j] = op(a_row[j*sa]);
                        }
                    }

                    out_row += n;
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

void exp(const Tensor &a, Tensor &out) {
    unary_elementwise(a, out, "exp", [](float x) { return std::exp(x); });
}

void log(const Tensor &a, Tensor &out) {
    unary_elementwise(a, out, "log", [](float x) { return std::log(x); });
}

/* ------------------------------------------------------------------------ */
/* Accumulate and reduce                                                     */
/* ------------------------------------------------------------------------ */

void add_into(Tensor &dst, const Tensor &src) {
    require_same_shape(dst, src, "add_into");
    require_contiguous_out(dst, "add_into");

    Walk ws = padded(src);

    const float *src_ptr = src.data();
    float *dst_row = dst.data();            // contiguous: rows are adjacent

    const std::size_t n = ws.shape[3];      // elements in one row
    const std::size_t ss = ws.strides[3];   // step along a row of src

    for (std::size_t i0 = 0; i0 < ws.shape[0]; i0++) {
        for (std::size_t i1 = 0; i1 < ws.shape[1]; i1++) {
            for (std::size_t i2 = 0; i2 < ws.shape[2]; i2++) {
                const float *src_row = src_ptr + row_start(ws, i0, i1, i2);

                if (ss == 1) {
                    // A contiguous run: the vectorisable case.
                    for (std::size_t j = 0; j < n; j++) {
                        dst_row[j] += src_row[j];
                    }
                } else {
                    for (std::size_t j = 0; j < n; j++) {
                        dst_row[j] += src_row[j*ss];
                    }
                }

                dst_row += n;
            }
        }
    }
}

void sum_into(Tensor &dst, const Tensor &src) {
    // Rank first - every shape(d) below asserts d < rank.
    if (dst.rank() != src.rank()) {
        throw std::invalid_argument("sum_into: rank mismatch");
    }

    // An extent either matches src or is 1, which marks a collapsed dimension.
    // Anything else is not a collapse.
    for (std::size_t d = 0; d < dst.rank(); ++d) {
        std::size_t ds = dst.shape(d);
        std::size_t ss = src.shape(d);
        if (ds != ss && ds != 1) {
            throw std::invalid_argument("sum_into: incompatible shape");
        }
    }

    require_contiguous_out(dst, "sum_into");

    Walk ws = padded(src);
    Walk wd = padded(dst);

    // Zero the stride on each collapsed dimension, so every index along it
    // lands on the same float and the += below does the summing. This is the
    // one place sum_into differs from add_into, which walks its destination
    // with a counter because its shapes always match.
    for (std::size_t d = 0; d < MAX_RANK; d++) {
        if (wd.shape[d] == 1) {
            wd.strides[d] = 0;
        }
    }

    float *dst_ptr = dst.data();
    const float *src_ptr = src.data();

    for (std::size_t i0 = 0; i0 < ws.shape[0]; i0++) {
        for (std::size_t i1 = 0; i1 < ws.shape[1]; i1++) {
            for (std::size_t i2 = 0; i2 < ws.shape[2]; i2++) {
                for (std::size_t i3 = 0; i3 < ws.shape[3]; i3++) {
                    std::size_t src_off = i0*ws.strides[0] + i1*ws.strides[1] + i2*ws.strides[2] + i3*ws.strides[3];
                    std::size_t dst_off = i0*wd.strides[0] + i1*wd.strides[1] + i2*wd.strides[2] + i3*wd.strides[3];
                    dst_ptr[dst_off] += src_ptr[src_off];
                }
            }
        }
    }
}

void reduce_max(const Tensor &a, Tensor &out) {
    // Rank first - every shape(d) below asserts d < rank.
    if (out.rank() != a.rank()) {
        throw std::invalid_argument("reduce_max: rank mismatch");
    }

    // Same rule as sum_into: an extent matches, or is 1 and collapses.
    for (std::size_t d = 0; d < out.rank(); ++d) {
        std::size_t os = out.shape(d);
        std::size_t as = a.shape(d);
        if (os != as && os != 1) {
            throw std::invalid_argument("reduce_max: incompatible shape");
        }
    }

    require_contiguous_out(out, "reduce_max");

    float *out_ptr = out.data();
    const float *a_ptr = a.data();

    // Seed with -inf, not zero. Every slot sees at least one element, so none
    // of these survive.
    const std::size_t n = out.numel();
    for (std::size_t i = 0; i < n; i++) {
        out_ptr[i] = -std::numeric_limits<float>::infinity();
    }

    Walk wa = padded(a);
    Walk wo = padded(out);

    // Stride 0 on each collapsed dimension, so every index along it lands on
    // the same float. Same trick as sum_into.
    for (std::size_t d = 0; d < MAX_RANK; d++) {
        if (wo.shape[d] == 1) {
            wo.strides[d] = 0;
        }
    }

    for (std::size_t i0 = 0; i0 < wa.shape[0]; i0++) {
        for (std::size_t i1 = 0; i1 < wa.shape[1]; i1++) {
            for (std::size_t i2 = 0; i2 < wa.shape[2]; i2++) {
                for (std::size_t i3 = 0; i3 < wa.shape[3]; i3++) {
                    std::size_t a_off = i0*wa.strides[0] + i1*wa.strides[1] + i2*wa.strides[2] + i3*wa.strides[3];
                    std::size_t out_off = i0*wo.strides[0] + i1*wo.strides[1] + i2*wo.strides[2] + i3*wo.strides[3];

                    if (a_ptr[a_off] > out_ptr[out_off]) {
                        out_ptr[out_off] = a_ptr[a_off];
                    }
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
