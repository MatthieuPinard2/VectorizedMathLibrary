#pragma once

#include <cstdlib>
#include <utility>
#include <intrin.h>
#include <immintrin.h>
#include <cassert>
#include <exception>

#if defined(_MSC_VER)
#define aligned_malloc _aligned_malloc
#define aligned_realloc _aligned_realloc
#define aligned_free _aligned_free
#else
#define __forceinline inline
#endif

// Abstraction layer.
namespace {
	constexpr size_t alignment = 64;
	constexpr size_t fpu_size = 64;
	__forceinline void prefetchL1(const __m256d* ptr) {
		_mm_prefetch((char*)ptr, _MM_HINT_T0);
	}
	template <class Real> 
	__forceinline size_t get_block_size(const size_t size) {
		const size_t block_size = (((sizeof(Real) * size) / fpu_size) + 1) * fpu_size;
		assert(size && block_size >= size * sizeof(Real) && block_size % fpu_size == 0);
		return block_size;
	}
	// Copies [begin, end[ into [begin_out, begin_out + (end - begin)[, 512b per pass.
	__forceinline void _memcpy(const void* begin, const void* end, void* begin_out) {
		assert(begin && end > begin && begin_out);
		assert(begin_out > end || ptrdiff_t(begin_out) + (ptrdiff_t(end) - ptrdiff_t(begin)) < ptrdiff_t(begin));
		__m256d* dest = (__m256d*)begin_out;
		const __m256d* src = (__m256d*)begin;
		while (src < (__m256d*)end) {
			prefetchL1(src);
			_mm256_stream_pd((double*)dest, _mm256_load_pd((double*)src));
			_mm256_stream_pd((double*)(dest + 1), _mm256_load_pd((double*)(src + 1)));
			dest += 2; src += 2;
		}
	}
};

template<class Real> class Vector {
private:
	Real* m_data;
	size_t m_size;
	Real* m_end;
public:
	// Returns the number of elements in the vector.
	size_t size() const noexcept {
		return m_size;
	}
	// Returns the maximum number of elements that can be stored in the vector.
	size_t capacity() const noexcept {
		const size_t capacity = ((Real*)m_end - (Real*)m_data);
		assert(m_end >= m_data && capacity > m_size);
		return capacity;
	}
	// Accessors 
	Real& operator[](const size_t i) {
		assert(i < m_size);
		return *(m_data + i);
	}
	Real const& operator[](const size_t i) const {
		assert(i < m_size);
		return *(m_data + i);
	}
	// Default constructor.
	Vector() noexcept : m_size(0), m_data(nullptr), m_end(nullptr) {}
	Vector(const size_t size) : m_size(size) {
		const size_t mem_size = get_block_size<Real>(size);
		m_data = (Real*)aligned_malloc(mem_size, alignment);
		if (!m_data)
			throw std::bad_alloc();
		m_end = (Real*)((intptr_t)m_data + mem_size);
		assert(m_end > m_data);
	}
	// Move constructors
	Vector<Real>& operator=(Vector<Real>&& other) noexcept {
		if (this != &other) {
			aligned_free(m_data);
			m_data = std::exchange(other.m_data, nullptr);
			m_size = std::exchange(other.m_size, 0);
			m_end = std::exchange(other.m_end, nullptr);
		}
		return *this;
	}
	Vector(Vector<Real>&& other) noexcept {
		m_data = std::exchange(other.m_data, nullptr);
		m_size = std::exchange(other.m_size, 0);
		m_end = std::exchange(other.m_end, nullptr);
	}
	// Copy constructor.
	Vector(const Vector<Real>& other) {
		const size_t mem_size = get_block_size<Real>(other.m_size);
		m_data = (Real*)aligned_malloc(mem_size, alignment);
		if (!m_data)
			throw std::bad_alloc();
		m_size = other.m_size;
		m_end = (Real*)((intptr_t)m_data + mem_size);
		assert(m_end > m_data);
		_memcpy(other.m_data, other.m_end, m_data);
	}
	Vector<Real>& operator=(const Vector<Real>& other) {
		if (this != &other) {
			const size_t mem_size = get_block_size<Real>(other.m_size);
			if (capacity() < other.size()) {
				m_data = (Real*)aligned_realloc(m_data, mem_size, alignment);
				if (!m_data)
					throw std::bad_alloc();
			}
			m_size = other.m_size;
			m_end = (Real*)((intptr_t)m_data + mem_size);
			assert(m_end > m_data);
			_memcpy(other.m_data, other.m_end, m_data);
		}
		return *this;
	}
	// Destructor
	~Vector() {
		aligned_free(m_data);
	}
	// Assignment Operators.
#define DEFINE_ASSIGNMENT_OPERATOR(op, intrinsic)                                            \
	__forceinline Vector<double>& operator##op(const double _Right) noexcept {               \
		auto lhs = (__m256d*)m_data;                                                         \
		const auto rhs = _mm256_broadcast_sd(&_Right);                                       \
		assert(lhs && m_end > (Real*)lhs);                                                   \
		while (lhs < (__m256d*)m_end) {                                                      \
			prefetchL1(lhs);                                                                 \
			const auto ymm1 = _mm256_load_pd((double*)lhs);                                  \
			const auto ymm2 = _mm256_load_pd((double*)(lhs + 1));                            \
			const auto ymm3 = ##intrinsic(ymm1, rhs);                                        \
			const auto ymm4 = ##intrinsic(ymm2, rhs);                                        \
			_mm256_store_pd((double*)lhs, ymm3);                                             \
			_mm256_store_pd((double*)(lhs + 1), ymm4);                                       \
			lhs += 2;                                                                        \
		}                                                                                    \
		return *this;                                                                        \
	}                                                                                        \
	__forceinline Vector<double>& operator##op(Vector<double> const& _Right) noexcept {      \
		auto lhs = (__m256d*)m_data;                                                         \
		auto rhs = (const __m256d*)_Right.m_data;                                            \
		assert(lhs && rhs && m_end > (Real*)lhs);                                            \
		assert(m_size == _Right.m_size);                                                     \
		while (lhs < (__m256d*)m_end) {                                                      \
			prefetchL1(rhs);                                                                 \
			const auto ymm1 = _mm256_load_pd((double*)rhs);                                  \
			const auto ymm2 = _mm256_load_pd((double*)(rhs + 1));                            \
			prefetchL1(lhs);                                                                 \
			const auto ymm3 = _mm256_load_pd((double*)lhs);                                  \
			const auto ymm4 = _mm256_load_pd((double*)(lhs + 1));                            \
			const auto ymm5 = ##intrinsic(ymm3, ymm1);                                       \
			const auto ymm6 = ##intrinsic(ymm4, ymm2);                                       \
			_mm256_store_pd((double*)lhs, ymm5);                                             \
			_mm256_store_pd((double*)(lhs + 1), ymm6);                                       \
			lhs += 2; rhs += 2;                                                              \
		}                                                                                    \
		return *this;                                                                        \
	}                                                                                        
#define DEFINE_ASSIGNMENT_OPERATOR_NON_COMMUTATIVE(op, inverted_op, intrinsic)               \
	DEFINE_ASSIGNMENT_OPERATOR(##op, ##intrinsic);                                           \
	__forceinline Vector<double>& ##inverted_op(const double _Right) noexcept {              \
		auto lhs = (__m256d*)m_data;                                                         \
		const auto rhs = _mm256_broadcast_sd(&_Right);                                       \
		assert(lhs && m_end > (Real*)lhs);                                                   \
		while (lhs < (__m256d*)m_end) {                                                      \
			prefetchL1(lhs);                                                                 \
			const auto ymm1 = _mm256_load_pd((double*)lhs);                                  \
			const auto ymm2 = _mm256_load_pd((double*)(lhs + 1));                            \
			const auto ymm3 = ##intrinsic(rhs, ymm1);                                        \
			const auto ymm4 = ##intrinsic(rhs, ymm2);                                        \
			_mm256_store_pd((double*)lhs, ymm3);                                             \
			_mm256_store_pd((double*)(lhs + 1), ymm4);                                       \
			lhs += 2;                                                                        \
		}                                                                                    \
		return *this;                                                                        \
	}                                                                                        \
	__forceinline Vector<double>& ##inverted_op(Vector<double> const& _Right) noexcept {     \
		auto lhs = (__m256d*)m_data;                                                         \
		auto rhs = (const __m256d*)_Right.m_data;                                            \
		assert(lhs && rhs && m_end > (Real*)lhs);                                            \
		assert(m_size == _Right.m_size);                                                     \
		while (lhs < (__m256d*)m_end) {                                                      \
			prefetchL1(rhs);                                                                 \
			const auto ymm1 = _mm256_load_pd((double*)rhs);                                  \
			const auto ymm2 = _mm256_load_pd((double*)(rhs + 1));                            \
			prefetchL1(lhs);                                                                 \
			const auto ymm3 = _mm256_load_pd((double*)lhs);                                  \
			const auto ymm4 = _mm256_load_pd((double*)(lhs + 1));                            \
			const auto ymm5 = ##intrinsic(ymm1, ymm3);                                       \
			const auto ymm6 = ##intrinsic(ymm2, ymm4);                                       \
			_mm256_store_pd((double*)lhs, ymm5);                                             \
			_mm256_store_pd((double*)(lhs + 1), ymm6);                                       \
			lhs += 2; rhs += 2;                                                              \
		}                                                                                    \
		return *this;                                                                        \
	}

	DEFINE_ASSIGNMENT_OPERATOR(+=, _mm256_add_pd);
	DEFINE_ASSIGNMENT_OPERATOR_NON_COMMUTATIVE(-=, inv_sub, _mm256_sub_pd);
	DEFINE_ASSIGNMENT_OPERATOR(*=, _mm256_mul_pd);
	DEFINE_ASSIGNMENT_OPERATOR_NON_COMMUTATIVE(/=, inv_div, _mm256_div_pd);
	
	// Unary operator+() : it does nothing.
	__forceinline Vector<double> const& operator+() const noexcept {
		return *this;
	}
	__forceinline Vector<double>& operator+() noexcept {
		return *this;
	}
	// Unary operator-().
	__forceinline Vector<double>& negate() noexcept {
		auto lhs = (__m256d*)m_data;
		const auto minusZero = _mm256_set1_pd(-0.0);
		assert(lhs && m_end > (Real*)lhs);
		while (lhs < (__m256d*)m_end) {
			prefetchL1(lhs);
			const auto ymm1 = _mm256_load_pd((double*)lhs);
			const auto ymm2 = _mm256_load_pd((double*)(lhs + 1));
			const auto ymm3 = _mm256_xor_pd(ymm1, minusZero);
			const auto ymm4 = _mm256_xor_pd(ymm2, minusZero);
			_mm256_store_pd((double*)lhs, ymm3);
			_mm256_store_pd((double*)(lhs + 1), ymm4);
			lhs += 2;
		}
		return *this;
	}
	
	// Fill from a scalar.
	__forceinline Vector<double>& fill(const double _Right) noexcept {
		auto lhs = (__m256d*)m_data;
		const auto rhs = _mm256_broadcast_sd(&_Right);
		assert(lhs && m_end > (Real*)lhs);
		while (lhs < (__m256d*)m_end) {
			_mm256_stream_pd((double*)lhs, rhs);
			_mm256_stream_pd((double*)(lhs + 1), rhs);
			lhs += 2;
		}
		return *this;
	}
};

// Definition of arithmetic operators (e.g. +, -, *, /)
// inverted_op is the inverted operator, i.e. inverted_op(x, y) = op(y, x)
#define DEFINE_OPERATOR(op, inverted_op)                             \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		const Vector<Real>& _Left, const Vector<Real>& _Right) {     \
		return std::move(Vector<Real>(_Left) ##op= _Right);          \
	}                                                                \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		const Vector<Real>& _Left, Vector<Real>&& _Right) {          \
		return std::move(_Right.##inverted_op(_Left));               \
	}                                                                \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		Vector<Real>&& _Left, const Vector<Real>& _Right) {          \
		return std::move(_Left ##op= _Right);                        \
	}                                                                \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		Vector<Real>&& _Left, Vector<Real>&& _Right) {               \
		return std::move(_Left ##op= _Right);                        \
	}                                                                \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		const double _Left, const Vector<Real>& _Right) {            \
		return std::move(Vector<Real>(_Right).##inverted_op(_Left)); \
	}                                                                \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		const double _Left, Vector<Real>&& _Right) {                 \
		return std::move(_Right.##inverted_op(_Left));               \
	}                                                                \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		const Vector<Real>& _Left, const double _Right) {            \
		return std::move(Vector<Real>(_Left) ##op= _Right);          \
	}                                                                \
	template <class Real>                                            \
	__forceinline constexpr Vector<Real> operator##op(               \
		Vector<Real>&& _Left, const double _Right) {                 \
		return std::move(_Left ##op= _Right);                        \
	}

DEFINE_OPERATOR(+, operator+=)
DEFINE_OPERATOR(*, operator*=)
DEFINE_OPERATOR(-, inv_sub)
DEFINE_OPERATOR(/ , inv_div)

// Unary minus.
template <class Real>
__forceinline constexpr Vector<Real> operator-(Vector<Real>& _Left) {     
	return std::move(Vector<Real>(_Left).negate());
}
template <class Real>
__forceinline constexpr Vector<Real> operator-(Vector<Real>&& _Left) {
	return std::move(_Left.negate());
}

// TODO : FMA.