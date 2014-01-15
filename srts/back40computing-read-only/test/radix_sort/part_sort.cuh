
// Sorting includes
#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/multi_buffer.cuh>
#include <sys/time.h>

template<int START_BIT, int NO_BITS>
int* doSortDevice(int *d_keys, int *d_values, int num_elements, bool keys_only);


template<int START_BIT, int NO_BITS>
timeval doSort(int *h_keys, int *h_values, int num_elements, bool keys_only)
{
	typedef int KeyType;
	typedef int ValueType;
	
	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	ValueType *d_values;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements);

	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice);

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	timeval before, after;
	
	gettimeofday(&before, NULL);

	int *valuesPtr = doSortDevice<START_BIT, NO_BITS>(d_keys, d_values, num_elements, keys_only);

	gettimeofday(&after, NULL);

	if (!keys_only)
	{
		// Copy out values
                cudaMemcpy(h_values, valuesPtr, sizeof(ValueType) * num_elements, cudaMemcpyDeviceToHost);

	}
	cudaFree(d_keys);
	cudaFree(d_values);

	timeval result;
	result.tv_sec = after.tv_sec - before.tv_sec;
	result.tv_usec = after.tv_usec - before.tv_usec;
	return result;
}

template<int START_BIT, int NO_BITS>
int* doSortDevice(int *d_keys, int *d_values, int num_elements, bool keys_only)
{
	typedef int KeyType;
	typedef int ValueType;

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	if (keys_only) {

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<KeyType> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(KeyType) * num_elements);

		// Sort
//		enactor.Sort(double_buffer, num_elements);
		enactor.OneRunSort<START_BIT, NO_BITS>(double_buffer, num_elements);
		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}
		return NULL;

	} else {

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<KeyType, ValueType> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;
		double_buffer.d_values[double_buffer.selector] = d_values;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(KeyType) * num_elements);
		cudaMalloc((void**) &double_buffer.d_values[double_buffer.selector ^ 1], sizeof(ValueType) * num_elements);

		// Sort
//		enactor.Sort(double_buffer, num_elements);
		enactor.OneRunSort<START_BIT, NO_BITS>(double_buffer, num_elements);


		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}
		if (double_buffer.d_values[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_values[double_buffer.selector ^ 1]);
		}
		return double_buffer.d_values[double_buffer.selector];
	}
}
