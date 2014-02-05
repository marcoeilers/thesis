#include <part_sort.cuh>
#include <sys/time.h>
#include <scatterKernel.cuh>
           
#define BITS 26
#define EXP 26
#define START (EXP - BITS)
  

template<int START_BIT, int NO_BITS>
std::pair<timeval, timeval> sortScatter(int *h_keys, int *h_values, int *h_result, int num_elements, bool keys_only)
{
	typedef int KeyType;
	typedef int ValueType;
	
	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	ValueType *d_values;
	int *d_result;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements);

	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice);

	timeval before, between, after;
	
	gettimeofday(&before, NULL);



//++++++ doSortDevice

	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

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


	cudaThreadSynchronize();
//printf("error after sort %i\n", cudaGetLastError());

        gettimeofday(&between, NULL);

        cudaMalloc((void**) &d_result, sizeof(int) * num_elements);
//printf("error after malloc %i\n", cudaGetLastError());
        scatter(d_keys, double_buffer.d_values[double_buffer.selector], d_result, num_elements);
//printf("error after scatter %i\n", cudaGetLastError());
        cudaThreadSynchronize();
        gettimeofday(&after, NULL);

		// Cleanup "pong" storage
		if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
		}
		if (double_buffer.d_values[double_buffer.selector ^ 1]) {
			cudaFree(double_buffer.d_values[double_buffer.selector ^ 1]);
		}


//------ doSortDevice
/*


	int *valuesPtr = doSortDevice<START_BIT, NO_BITS>(d_keys, d_values, num_elements, keys_only);
	cudaThreadSynchronize();
printf("error after sort %i\n", cudaGetLastError());

	gettimeofday(&between, NULL);

	cudaMalloc((void**) &d_result, sizeof(int) * num_elements);
printf("error after malloc %i\n", cudaGetLastError());
	scatter(d_keys, valuesPtr, d_result, num_elements);
printf("error after scatter %i\n", cudaGetLastError());
	cudaThreadSynchronize();
//int error = cudaGetLastError();
//printf("error after scatter %i\n", error);
	gettimeofday(&after, NULL);
*/
	cudaMemcpy(h_result, d_result, sizeof(int) * num_elements, cudaMemcpyDeviceToHost);


	cudaFree(d_keys);
	cudaFree(d_values);

	timeval result1, result2;
	result2.tv_sec = after.tv_sec - between.tv_sec;
	result2.tv_usec = after.tv_usec - between.tv_usec;
	result1.tv_sec = between.tv_sec - before.tv_sec;
	result1.tv_usec = between.tv_usec - before.tv_usec;
	std::pair<timeval, timeval> result(result1, result2);
	return result;
}

int main()
{
	printf("rand max is %i\n", RAND_MAX);
	typedef int KeyType;
	typedef int ValueType;

	unsigned int num_elements = 1 << EXP;


        // Allocate host problem data
	KeyType *h_keys = new KeyType[num_elements];
	ValueType *h_values = new ValueType[num_elements];
	int *h_result = new int[num_elements];

        // Initialize host problem data

        for (int i = 0; i < num_elements; ++i)
        {
                h_keys[i] = rand() % num_elements;
                h_values[i] = i;
        }
	std::pair<timeval, timeval> result;
	result = sortScatter<START, BITS>(h_keys, h_values, h_result, num_elements, false);
	float time1 = result.first.tv_sec * 1e9 + result.first.tv_usec * 1e3;
	float time2 = result.second.tv_sec * 1e9 + result.second.tv_usec * 1e3;
	printf("%i\t%i\t%f\t%f\t%f\n", num_elements, BITS, time1 / num_elements, time2 / num_elements, (time1 + time2) / num_elements);

	delete(h_keys);
        delete(h_values);
	delete(h_result);
	return 0;
}
