
// Sorting includes
#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/multi_buffer.cuh>

#include <sys/time.h>

template <int START_BIT, int NO_BITS>
timeval one_run_sort(int *h_keys, int *h_values, int num_elements, bool keys_only)
{
	int *d_keys;
	int *d_values;
	cudaMalloc((void**) &d_keys, sizeof(int) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(int) * num_elements);

	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

	if (!keys_only)
		cudaMemcpy(d_values, h_values, sizeof(int) * num_elements, cudaMemcpyHostToDevice);

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	timeval before, after;

	if (keys_only) {

		// Create ping-pong storage wrapper
		b40c::util::DoubleBuffer<int> double_buffer;

		// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
		double_buffer.d_keys[double_buffer.selector] = d_keys;

		// Allocate pong buffer
		cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(int) * num_elements);

		gettimeofday(&before, NULL);
		// Sort
		enactor.Sort(double_buffer, num_elements);
		cudaThreadSynchronize();
		gettimeofday(&after, NULL);

		cudaMemcpy(h_keys, double_buffer.d_keys[double_buffer.selector], sizeof(int) * num_elements, cudaMemcpyDeviceToHost);

		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}

	} else {

                // Create ping-pong storage wrapper
                b40c::util::DoubleBuffer<int, int> double_buffer;

                // The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
                double_buffer.d_keys[double_buffer.selector] = d_keys;
                double_buffer.d_values[double_buffer.selector] = d_values;

                // Allocate pong buffer
                cudaMalloc((void**) &double_buffer.d_keys[double_buffer.selector ^ 1], sizeof(int) * num_elements);
                cudaMalloc((void**) &double_buffer.d_values[double_buffer.selector ^ 1], sizeof(int) * num_elements);

		gettimeofday(&before, NULL);
                // Sort
                enactor.Sort(double_buffer, num_elements);
		cudaThreadSynchronize();
		gettimeofday(&after, NULL);

		// Copy out keys & values
		cudaMemcpy(h_keys, double_buffer.d_keys[double_buffer.selector], sizeof(int) * num_elements, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_values, double_buffer.d_values[double_buffer.selector], sizeof(int) * num_elements, cudaMemcpyDeviceToHost);

                // Cleanup "pong" storage
                if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
                        cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
                }
                if (double_buffer.d_values[double_buffer.selector ^ 1]) {
                        cudaFree(double_buffer.d_values[double_buffer.selector ^ 1]);
                }

        }

        cudaFree(d_keys);
        cudaFree(d_values);

	timeval result;
	result.tv_sec = after.tv_sec - before.tv_sec;
	result.tv_usec = after.tv_usec - before.tv_usec;
	return result;
}

