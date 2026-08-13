#ifndef TENSOR_H
#define TENSOR_H

#include "aligned.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>

/*
 Tensor - shape, strides and an offset into a refcounted buffer.

 Row-major throughout, float32 only. Element (i0, i1, ...) lives at
 data()[i0*stride(0) + i1*stride(1) + ...], where data() already has the
 offset folded in.

 Three deliberate divergences from PyTorch, all of them to keep the kernels
 free of special cases:
   - minimum rank 1; a scalar is shape {1}, there is no rank-0 tensor
   - no zero-length dimensions, so there are no empty tensors
   - reshape throws on a non-contiguous tensor rather than silently copying
*/

// A constexpr rather than a #define: ALIGNMENT_REQ is a macro only because it
// arrived from C, and new code should not add to that. This one is typed,
// scoped, and visible to the debugger.
//
// 4 covers everything on the roadmap. Shape and strides are inline arrays of
// this length, so a Tensor costs no heap allocation beyond its buffer - which
// matters once phase 8 constructs one per operation per training step.
constexpr std::size_t MAX_RANK = 4;

static_assert(MAX_RANK >= 2, "GEMM operands are rank 2");

// Refcounted buffer. Owns storage and nothing else: no shape, no strides, no
// opinion about how the floats are indexed.
//
// Sharing happens through shared_ptr<Storage> held by Tensor, not by copying
// Storage itself - hence copy stays deleted here, as it was on the owning
// rank-2 matrix type this replaced.
class Storage {
public:
    // Allocates size floats, zero-initialised, base pointer aligned to
    // ALIGNMENT_REQ.
    //
    // Throws std::invalid_argument if size is zero or would overflow when
    // converted to bytes; std::bad_alloc if the allocation fails.
    explicit Storage(std::size_t size);

    ~Storage() = default;

    // Neither copied nor moved. A Storage is constructed in place by
    // make_shared and owned by that shared_ptr for the rest of its life, so a
    // move would only ever be an untested route to a moved-from state that
    // nothing needs. Deleting it removes the state rather than documenting it.
    Storage(const Storage &)            = delete;
    Storage &operator=(const Storage &) = delete;
    Storage(Storage &&)                 = delete;
    Storage &operator=(Storage &&)      = delete;

    float       *data() noexcept;
    const float *data() const noexcept;

    // In floats, not bytes. The footprint assert on Tensor needs this.
    std::size_t size() const noexcept;

private:
    std::unique_ptr<float[], AlignedDeleter> data_;
    std::size_t                              size_ = 0;
};

class Tensor {
public:
    /* -------------------------------------------------------------------- */
    /* Runtime shapes                                                        */
    /*                                                                       */
    /* Four functions come in pairs - this constructor, permute, expand and  */
    /* reshape - an initializer_list overload for literal call sites and a   */
    /* std::span one for shapes only known at runtime.                       */
    /*                                                                       */
    /* std::initializer_list cannot be built at runtime: the compiler        */
    /* synthesises its backing array at the call site, and nothing lets you  */
    /* make one from a pointer and a length. A shape arriving from Python    */
    /* has no other way in. std::span IS that pointer and length, non-owning,*/
    /* binding from vector, array, C array or an initializer_list alike -    */
    /* the C idiom of (const T *, size_t) with a type on it.                 */
    /*                                                                       */
    /* Validation lives in the span overload. The initializer_list one       */
    /* forwards, so there is one copy of every check.                        */
    /*                                                                       */
    /* The pair is unambiguous permanently, not merely until span gains an   */
    /* initializer_list constructor in C++26. List-initialisation considers  */
    /* initializer_list constructors FIRST and stops once one is viable, so  */
    /* Tensor{2,3} keeps picking that one whatever span acquires later; and  */
    /* for a braced argument such as t.permute({0,1}) the conversion to      */
    /* initializer_list is ranked by its elements - an exact match here -    */
    /* while the one to span would be user-defined, which loses.             */
    /* -------------------------------------------------------------------- */

    // Allocates a contiguous row-major tensor of the given shape, zeroed.
    //
    // Throws std::invalid_argument on rank 0, on rank above MAX_RANK, on any
    // zero-length dimension, and on a shape whose element count overflows.
    explicit Tensor(std::initializer_list<std::size_t> shape);

    // Same contract, runtime shape. See the block above.
    explicit Tensor(std::span<const std::size_t> shape);

    // No default constructor on purpose. An empty default-constructed state
    // means every accessor grows an is-it-valid branch, and the moved-from
    // state is already one such state. Phase 5 wants std::optional<Tensor> for
    // a gradient slot anyway - "no gradient yet" is genuinely an optional.

    ~Tensor()                                  = default;
    Tensor(Tensor &&) noexcept                 = default;
    Tensor &operator=(Tensor &&) noexcept      = default;

    // Copy is SHALLOW and defaulted - the opposite of the rank-2 matrix type
    // this replaced, which deleted copy outright.
    // Two Tensors sharing one buffer is the whole point of the type: it is how
    // a view exists at all. clone() is the deep copy.
    //
    // The cost of this: the compiler no longer catches an accidental copy. A
    // 4 MB buffer that used to be a compile error is now a refcount bump.
    Tensor(const Tensor &)            = default;
    Tensor &operator=(const Tensor &) = default;

    std::size_t rank() const noexcept;

    // Both assert dim < rank() in debug and are undefined above it. Bounds
    // checks that vanish under -DNDEBUG, same as operator() below.
    std::size_t shape(std::size_t dim) const noexcept;
    std::size_t stride(std::size_t dim) const noexcept;

    // Product of the shape. Not the same as storage size once views exist:
    // a slice has fewer elements than the buffer behind it.
    std::size_t numel() const noexcept;

    // Storage base plus offset_, so indexing never has to add the offset again.
    float       *data() noexcept;
    const float *data() const noexcept;

    // Computed, never cached. At most MAX_RANK multiplies, and a cached flag is
    // a lie waiting to happen the first time a view forgets to clear it.
    bool is_contiguous() const noexcept;

    // True when both refer to the same Storage. Written for the tests: they
    // want to know about sharing, and comparing data() pointers gives a false
    // negative on two views sitting at different offsets.
    bool shares_storage_with(const Tensor &other) const noexcept;

    // Deep copy: fresh Storage, contiguous row-major, same shape and values.
    // Values are read through this tensor's strides, so cloning a transposed
    // view materialises the transpose.
    Tensor clone() const;

    // Both require is_contiguous() and throw std::invalid_argument otherwise.
    // Writing through a broadcast view would have several logical elements
    // aliasing one float, which is never what either of these means.
    void zero();
    void randomize();

    /* -------------------------------------------------------------------- */
    /* Views                                                                 */
    /*                                                                       */
    /* All six share this tensor's Storage and allocate nothing, with the one */
    /* stated exception in contiguous(). All are const and return a new       */
    /* Tensor: a view is a second handle onto one buffer, not a mutation of   */
    /* the handle you already had.                                            */
    /*                                                                       */
    /* All six report bad arguments by throwing std::invalid_argument rather  */
    /* than asserting. These take runtime values from a caller, unlike        */
    /* shape() and stride(), which assert because they are hot and their      */
    /* argument is nearly always a literal.                                   */
    /* -------------------------------------------------------------------- */

    // Swaps two dimensions. Shape and strides swap together; nothing moves in
    // memory, so the result is almost never contiguous.
    //
    // Throws if either dimension is out of range.
    Tensor transpose(std::size_t d0, std::size_t d1) const;

    // General dimension reordering: result dimension i takes its extent and
    // stride from this tensor's dimension dims[i].
    //
    // Throws unless dims is exactly a permutation of 0..rank()-1 - the same
    // length, every dimension present, none twice.
    Tensor permute(std::initializer_list<std::size_t> dims) const;

    // Same contract, runtime dims. See the runtime-shapes block above.
    Tensor permute(std::span<const std::size_t> dims) const;

    // Narrows one dimension to [start, start + count). Strides are untouched;
    // only the offset moves and one extent shrinks. This is the operation
    // offset_ exists for.
    //
    // Throws if dim is out of range, if count is 0 - there are no empty
    // tensors - or if start + count exceeds the extent. Write that last check
    // as start + count > shape, never start > shape - count, which wraps.
    Tensor slice(std::size_t dim, std::size_t start, std::size_t count) const;

    // Broadcasts: any dimension whose extent is 1 may be stretched to any
    // extent, and gets stride 0 so every index on it addresses one element.
    // Rank is unchanged; every other extent must match exactly.
    //
    // The result must never be written through - several logical elements
    // alias one float. That is why the step 5 op layer requires its output
    // contiguous, and why zero() and randomize() refuse a non-contiguous
    // tensor.
    //
    // Throws if the rank differs or a dimension whose extent is not 1 is
    // asked to change.
    Tensor expand(std::initializer_list<std::size_t> shape) const;

    // Same contract, runtime shape. See the runtime-shapes block above.
    Tensor expand(std::span<const std::size_t> shape) const;

    // Reinterprets the same elements under a new shape, with fresh row-major
    // strides. Requires the same element count.
    //
    // Throws unless is_contiguous(), deliberately: torch's reshape falls back
    // to a copy and its view throws, and one function with two performance
    // profiles behind identical syntax is how a training loop gets
    // mysteriously slow. Callers who want the copy write
    // .contiguous().reshape(...) and can see themselves paying for it.
    Tensor reshape(std::initializer_list<std::size_t> shape) const;

    // Same contract, runtime shape. See the runtime-shapes block above.
    Tensor reshape(std::span<const std::size_t> shape) const;

    // Returns a tensor with the same values in dense row-major order.
    //
    // The ONLY one of the six that may allocate, and only when it has to: if
    // this tensor is already contiguous it returns a handle onto the same
    // buffer and costs a refcount bump. Every kernel boundary calls this, so
    // that fast path is the reason Tensor is copyable at all.
    Tensor contiguous() const;

    // Element access. sizeof...(idx) must equal rank(); asserted in debug,
    // undefined otherwise. Defined in the header because it is a template.
    template <typename... Idx>
    float &operator()(Idx... idx) {
        static_assert(sizeof...(Idx) <= MAX_RANK, "too many indices for any Tensor");
        assert(sizeof...(idx) == rank_);

        std::array<std::size_t, MAX_RANK> indices{static_cast<std::size_t>(idx)...};

        std::size_t offset = 0;
        for (std::size_t d = 0; d < rank_; ++d) {
            offset += indices[d] * strides_[d];
        }

        return data()[offset];
    }

    template <typename... Idx>
    const float &operator()(Idx... idx) const {
        static_assert(sizeof...(Idx) <= MAX_RANK, "too many indices for any Tensor");
        assert(sizeof...(idx) == rank_);

        std::array<std::size_t, MAX_RANK> indices{static_cast<std::size_t>(idx)...};

        std::size_t offset = 0;
        for (std::size_t d = 0; d < rank_; ++d) {
            offset += indices[d] * strides_[d];
        }

        return data()[offset];
    }

private:
    // Validates a shape and packs it into a fixed-size array.
    //
    // Static because it has to run BEFORE any member exists: a delegating
    // constructor delegates from its mem-init list, so there is nowhere to put
    // a check that must happen first. This is the "validate, then initialise"
    // idiom - the same one that would let data_ be const if it ever wanted to be.
    //
    // Throws std::invalid_argument on rank 0, on rank above MAX_RANK, and on a
    // zero-length extent. Rank is checked before anything is copied, so the
    // array can never be overrun by an over-long list.
    static std::array<std::size_t, MAX_RANK>
    checked_shape(std::initializer_list<std::size_t> shape);

    // Same contract, runtime shape, and the one place every check lives: the
    // four public span overloads all reach it.
    //
    // The order above is load-bearing, not stylistic. std::copy(shape.begin(),
    // shape.end(), result.begin()) is the obvious first draft here and it
    // overruns the MAX_RANK array on a rank-5 input before any check has run -
    // in the function whose whole job is rejecting bad shapes, and in Release
    // too. Reject the rank, then the extents, then pack.
    static std::array<std::size_t, MAX_RANK>
    checked_shape(std::span<const std::size_t> shape);
    static std::size_t numel_of(const std::array<std::size_t, MAX_RANK>& shape, std::size_t rank);
    static std::array<std::size_t, MAX_RANK> row_major_strides(const std::array<std::size_t, MAX_RANK>& shape, std::size_t rank);

    // Allocating constructor: fresh Storage sized to the shape product,
    // row-major strides derived from the shape, offset 0.
    //
    // Takes a shape already known good - from checked_shape, or from another
    // Tensor's shape_ - so it asserts rather than throws on rank and extents.
    // It still throws if the element count overflows, because it has to
    // compute that count to size the allocation either way.
    //
    // This is what the public constructor delegates to, and what clone() and
    // step 4's contiguous() build with.
    Tensor(const std::array<std::size_t, MAX_RANK> &shape, std::size_t rank);

    // View constructor: shares an existing buffer. No allocation, and no
    // stride derivation - strides and offset are handed in, because a
    // transposed or broadcast view has neither row-major strides nor offset 0.
    //
    // This is the ONLY way to construct a view, deliberately. Every step 4
    // operation funnels through here, which is what makes
    // assert_within_storage() a guarantee rather than a convention: a view
    // that skipped the check would have to skip this constructor to do it.
    // Copying *this and mutating shape_/strides_/offset_ in place is the
    // shortcut that quietly gives that up.
    Tensor(std::shared_ptr<Storage> storage, std::size_t offset,
           const std::array<std::size_t, MAX_RANK> &shape,
           const std::array<std::size_t, MAX_RANK> &strides, std::size_t rank);

    // Debug-only invariant: the last element this tensor can address must lie
    // inside the buffer.
    //
    //   offset_ + Σ_d (shape_[d] - 1) * strides_[d]  <  storage_->size()
    //
    // This is the structural detector for bad offset arithmetic. ASan cannot
    // see a column overrun, because on every row but the last it lands in the
    // next row of the same allocation - but a wrong slice offset or a
    // mis-permuted stride blows this bound immediately.
    //
    // A dimension expanded to stride 0 contributes nothing to the sum, which
    // is correct rather than a gap: every index on it addresses one element.
    void assert_within_storage() const noexcept;

    std::shared_ptr<Storage>            storage_;
    std::size_t                         offset_ = 0; // in floats, from the base
    std::array<std::size_t, MAX_RANK>   shape_{};
    std::array<std::size_t, MAX_RANK>   strides_{};
    std::size_t                         rank_ = 0;
};

// Inverted from the equivalents on the type this replaced, and inverted on
// purpose. If you reach for
// = delete on the copy operations out of habit, the build stops here.
static_assert(std::is_copy_constructible_v<Tensor>,
              "Tensor copy must be shallow and allowed - it is how views exist");
static_assert(std::is_copy_assignable_v<Tensor>,
              "Tensor copy-assignment must be shallow and allowed");
static_assert(std::is_nothrow_move_constructible_v<Tensor>,
              "Tensor must be nothrow-movable so containers can relocate it");
static_assert(std::is_nothrow_move_assignable_v<Tensor>,
              "Tensor must be nothrow-move-assignable");

// Nothing in phase 3 needs a default-constructed Tensor, and phase 5 should
// reach for std::optional<Tensor> rather than an empty one.
static_assert(!std::is_default_constructible_v<Tensor>,
              "Tensor has no valid empty state - use std::optional<Tensor>");

/*
 Thread safety, stated precisely because it is easy to state loosely:

   - Two DISTINCT shared_ptr objects that share a control block may be copied
     and destroyed concurrently. The refcount is atomic; this is safe.
   - Concurrent access to the SAME shared_ptr object, one thread reading it
     while another assigns to it, is a data race like any other.
   - Neither protects the float data. Two Tensors sharing a buffer are two
     handles onto the same memory, and gemm_multithreaded already exists.
*/

#endif
