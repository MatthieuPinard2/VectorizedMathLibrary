#pragma once

#include <cstdlib>
#include <cstring>
#include <utility>
#include <intrin.h>
#include <cassert>
#include <iostream>

// Abstraction layer.
namespace {
	constexpr size_t alignment = 64;
	constexpr size_t fpu_size = 64;
	__forceinline void prefetchL1(const __m256d* ptr) {
		_mm_prefetch((char*)ptr, _MM_HINT_T0);
	}
	template <class Real> 
	__forceinline size_t get_block_size(const size_t size) {
		return (((sizeof(Real) * size) / fpu_size) + 1) * fpu_size;
	}
	// Copies [begin, end[ into [begin_out, begin_out + (end - begin)[, 512b per pass.
	__forceinline void _memcpy(const void* begin, const void* end, void* begin_out) {
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
	size_t m_size;
	Real* m_data;
	Real* m_end;
public:
	size_t size() const noexcept {
		return m_size;
	}
	size_t capacity() const noexcept {
		return get_block_size<Real>(m_size) / sizeof(Real);
	}
	Real& operator[](const size_t i) {
		return *(m_data + i);
	}
	Real const& operator[](const size_t i) const {
		return *(m_data + i);
	}
	// Default constructor.
	Vector() noexcept : m_size(0), m_data(nullptr), m_end(nullptr) {}
	Vector(const size_t size) : m_size(size) {
		const size_t mem_size = get_block_size<Real>(size);
		m_data = (Real*)_aligned_malloc(mem_size, alignment);
		m_end = (Real*)((char*)m_data + mem_size);
	}
	// Move constructors
	Vector<Real>& operator=(Vector<Real>&& other) noexcept {
		if (this != &other) {
			_aligned_free(m_data);
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
		m_size = other.m_size;
		const size_t mem_size = get_block_size<Real>(m_size);
		m_data = (Real*)_aligned_malloc(mem_size, alignment);
		m_end = (Real*)((char*)m_data + mem_size);
		_memcpy(other.m_data, other.m_end, m_data);
	}
	Vector<Real>& operator=(const Vector<Real>& other) {
		// Self-copy guard.
		if (this != &other) {
			const size_t mem_size = get_block_size<Real>(other.m_size);
			// We allocate again.
			if (capacity() < other.size()) {
				m_data = (Real*)_aligned_realloc(m_data, mem_size, alignment);
			}
			m_size = other.m_size;
			m_end = (Real*)((char*)m_data + mem_size);
			_memcpy(other.m_data, other.m_end, m_data);
		}
		return *this;
	}
	// Destructor
	~Vector() {
		_aligned_free(m_data);
		m_data = 0;
	}
	// Operations on vectors.
	// 1/ Fill from an input
	// 2/ operator+, operator-, operator*, operator/ (scalar, vector)
	// 3/ Masks
	__forceinline Vector<double>& operator+=(const double scalar) {
		__m256d* lhs = (__m256d*)m_data;
		const __m256d rhs = _mm256_broadcast_sd(&scalar);
		while (lhs < (__m256d*)m_end) {
			prefetchL1(lhs);
			*lhs = _mm256_add_pd(*lhs, rhs);
			++lhs;
		}
		return *this;
	}
	__forceinline Vector<double>& operator+=(Vector<double> const& vector) {
		__m256d* lhs = (__m256d*)m_data;
		const __m256d* rhs = (__m256d*)vector.m_data;
		assert(m_size == vector.m_size);
		/*while (lhs < (__m256d*)m_end) {
			prefetchL1(lhs);
			prefetchL1(rhs);
			*lhs = _mm256_add_pd(*lhs, *rhs);
			++lhs; ++rhs;
		}*/
		while (lhs < (__m256d*)m_end) {
			prefetchL1(rhs);
			const __m256d ymm2 = _mm256_load_pd((double*)rhs);
			const __m256d ymm4 = _mm256_load_pd((double*)(rhs + 1));
			prefetchL1(lhs);
			const __m256d ymm1 = _mm256_load_pd((double*)lhs);
			const __m256d ymm3 = _mm256_load_pd((double*)(lhs + 1));
			_mm256_stream_pd((double*)lhs, _mm256_add_pd(ymm1, ymm2));
			_mm256_stream_pd((double*)(lhs + 1), _mm256_add_pd(ymm3, ymm4));
			lhs += 2; rhs += 2;
		}
		return *this;
	}
	__forceinline Vector<double>& operator=(const double scalar) {
		__m256d* lhs = (__m256d*)m_data;
		const __m256d rhs = _mm256_broadcast_sd(&scalar);
		while (lhs < (__m256d*)m_end) {
			prefetchL1(lhs);
			*lhs = rhs;
			++lhs;
		}
		return *this;
	}
};

template <class Real>
__forceinline constexpr Vector<Real> operator+(
	const Vector<Real>& _Left, const Vector<Real>& _Right) {
	return std::move(Vector<Real>(_Left) += _Right);
}

template <class Real>
__forceinline constexpr Vector<Real> operator+(
	const Vector<Real>& _Left, Vector<Real>&& _Right) {
	// Because operator+ is commutative in R^n, we can write the statement below.
	return std::move(_Right += _Left);
	// return std::move(Vector<Real>(_Left) += _Right);
}

template <class Real>
__forceinline constexpr Vector<Real> operator+(
	Vector<Real>&& _Left, const Vector<Real>& _Right) {
	return std::move(_Left += _Right);
}

template <class Real>
__forceinline constexpr Vector<Real> operator+(
	Vector<Real>&& _Left, Vector<Real>&& _Right) {
	return std::move(_Left += _Right);
}

