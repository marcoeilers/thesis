#include <part_sort.cuh>
#include <sys/time.h>
#include <scatterKernel.cuh>

#define BIN_BITS 26
#define BITS 0
#define EXP 26
//#define START 0
#define START (BIN_BITS - BITS)
#define BUCKETS (1 << BIN_BITS)

void printDeviceArrays(int *d_array1, int *d_array2, int length)
{
	int *array1 = (int*) malloc(length * sizeof(int));
	int *array2 = (int*) malloc(length * sizeof(int));

	cudaMemcpy(array1, d_array1, length * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(array2, d_array2, length * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < length; i++)
	{
		printf("index %i first array %u second array %u\n", i, array1[i], array2[i]);
	}

	free(array2);
	free(array1);
}


void cpuMR(int *keys, int *values, int *result, int num_elements)
{
	for (int i = 0; i < BUCKETS; i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
		result[keys[i]] += values[i]; 
}

template<int START_BIT, int NO_BITS>
std::pair<timeval, timeval> sortMR(int *h_keys, int *h_values, int *h_result, int num_elements)
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
        int *d_double_keys, *d_double_values;

        cudaMalloc((void**) &d_double_keys, sizeof(KeyType) * num_elements);
        cudaMalloc((void**) &d_double_values, sizeof(ValueType) * num_elements);
        double_buffer.d_keys[double_buffer.selector ^ 1] = d_double_keys;
        double_buffer.d_values[double_buffer.selector ^ 1] = d_double_values;


	// Sort
//	enactor.Sort(double_buffer, num_elements);
	enactor.OneRunSort<START_BIT, NO_BITS>(double_buffer, num_elements);


	cudaThreadSynchronize();
//printf("error after sort %i\n", cudaGetLastError());

        gettimeofday(&between, NULL);

//	printf("\n\n\n\nkeys values after sort\n");
//	printDeviceArrays(d_keys, double_buffer.d_values[double_buffer.selector], 50);	

        cudaMalloc((void**) &d_result, sizeof(int) * BUCKETS);
	cudaMemset(d_result, 0, sizeof(int) * BUCKETS);

//printf("error after malloc %i\n", cudaGetLastError());
        multiReduce(double_buffer.d_keys[double_buffer.selector], double_buffer.d_values[double_buffer.selector], d_result, num_elements);
//printf("error after scatter %i\n", cudaGetLastError());
        cudaThreadSynchronize();
        gettimeofday(&after, NULL);

	// Cleanup "pong" storage
	if (d_double_keys) {
		cudaFree(d_double_keys);
	}
	if (d_double_values) {
		cudaFree(d_double_values);
	}

//	printf("\n\n\n\nvalues after multiReduce\n");
//	printDeviceArrays(d_result, d_result, 100);

	cudaMemcpy(h_result, d_result, sizeof(int) * BUCKETS, cudaMemcpyDeviceToHost);


	cudaFree(d_keys);
	cudaFree(d_values);
	cudaFree(d_result);

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
	int *h_result = new int[BUCKETS];
	int *result_cpu = new int[BUCKETS];

        // Initialize host problem data
	int index = rand() % BUCKETS;
        for (int i = 0; i < num_elements; ++i)
        {
		h_keys[i] = index;
//                h_keys[i] = rand() % BUCKETS;
                h_values[i] = rand() % 100;
// 		if (i < 50)
//		printf("original key value %i %i %i\n", i, h_keys[i], h_values[i]);
        }
	std::pair<timeval, timeval> result;
	result = sortMR<START, BITS>(h_keys, h_values, h_result, num_elements);
	
	cpuMR(h_keys, h_values, result_cpu, num_elements);


	bool correct = true;
	for (int i = 0; i < BUCKETS; i++)
	{
		if (h_result[i] != result_cpu[i])
		{
			printf("i sense a discrepancy! %i %i %i\n", i, h_result[i], result_cpu[i]);
			correct = false;
		}
	}
	if (correct)
	printf("correct result!\n");

	printf("last error is %i\n", cudaGetLastError());

	float time1 = result.first.tv_sec * 1e9 + result.first.tv_usec * 1e3;
	float time2 = result.second.tv_sec * 1e9 + result.second.tv_usec * 1e3;
	printf("%i\t%i\t%f\t%f\t%f\n", num_elements, BITS, time1 / num_elements, time2 / num_elements, (time1 + time2) / num_elements);

	delete(result_cpu);
	delete(h_keys);
        delete(h_values);
	delete(h_result);
	return 0;
}
