#include "Vector.h"
#include <iostream>
#include <chrono>
#include <Windows.h>

template <class T>
void print(Vector<T> const& x) {
	std::cout << "{ ";
	for (int i = 0; i < x.size(); ++i) {
		std::cout << x[i];
		if (i != x.size() - 1) {
			std::cout << ", ";
		}
	}
	std::cout << " }" << std::endl;
}

int main() {
	unsigned __int64 affinityMask = 1 << 3;
	SetProcessAffinityMask(GetCurrentProcess(), affinityMask);
	SetThreadAffinityMask(GetCurrentThread(), affinityMask);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	auto begin = std::chrono::high_resolution_clock::now();
	size_t tries = 500;
	double x = 0.0;
	for (size_t i = 0; i < tries; ++i) {
		Vector<double> a(2000000);
		Vector<double> b(2000000);
		Vector<double> c(2000000);
		Vector<double> d(2000000);
		// a = 5.0
		a.fill(5.0);
		// b = 10.0
		b.fill(7.0);
		b += 3.0;
		// c = 15.0
		c = a;
		c += b;
		// d = 0.0
		d.fill(0.0);
		// Chained operator+ = 5 + 10 + 15 + 0 = 30.0
		a = a + b + c + (d * 4.0);
		a = a + (b + (c + d));
		a = 2.0 * (a / 2.0);
		a = +a;
		x += a[150000];
		x += a[0];
		x += a[2000000 - 1];
	}
	std::cout << (x - tries * 165.0) << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << (end - begin).count() / 1000000 << std::endl;
	return 0;
}