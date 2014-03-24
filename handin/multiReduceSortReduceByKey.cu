#include <part_sort.cuh>
#include <sys/time.h>
#include "kernels/reducebykey.cuh"

using namespace mgpu;

#define PREPROCESS true
#define BIN_BITS 6
#define EXP 26
#define BUCKETS (1 << BIN_BITS)
#define NO_RUNS 8


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

std::pair<timeval, timeval> add(std::pair<timeval, timeval> a, std::pair<timeval, timeval> b)
{
        std::pair<timeval, timeval> result;
        result.first.tv_sec = a.first.tv_sec + b.first.tv_sec;
        result.second.tv_sec = a.second.tv_sec + b.second.tv_sec;
        result.first.tv_usec = a.first.tv_usec + b.first.tv_usec;
        result.second.tv_usec = a.second.tv_usec + b.second.tv_usec;
        return result;
}


void cpuMR(int *keys, int *values, int *result, int num_elements)
{
	for (int i = 0; i < BUCKETS; i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
		result[keys[i]] += values[i]; 
}

template<typename KeyType, typename ValType>
void simpleReduceByKey(CudaContext& context, KeyType *indices, ValType *values, int num_elements, int buckets, int *numSegments, int *d_bucketLabels, int *d_bucketValues)
{	
	std::auto_ptr<ReduceByKeyPreprocessData> preprocessData;

		ReduceByKeyPreprocess<ValType>(num_elements, indices, 
			d_bucketLabels, mgpu::equal_to<KeyType>(),
			numSegments, (int*)0, &preprocessData, context);

		ReduceByKeyApply(*preprocessData, values, (ValType)0, 
			mgpu::plus<ValType>(), d_bucketValues, context);
}


std::pair<timeval, timeval> sortMR(int *h_keys, int *h_values, int *h_result, int num_elements)
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

	timeval before, between, after;
	
	gettimeofday(&before, NULL);


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
	enactor.OneRunSort<0, BIN_BITS>(double_buffer, num_elements);


	cudaThreadSynchronize();

        gettimeofday(&between, NULL);

	int *valuesPtr = double_buffer.d_values[double_buffer.selector];
	int *keysPtr = double_buffer.d_keys[double_buffer.selector];

//	int *valuesPtr = ((BIN_BITS + 4) / 5) % 2 == 0 ? double_buffer.d_values[double_buffer.selector] : double_buffer.d_values[double_buffer.selector ^ 1];
//	int *keysPtr = ((BIN_BITS + 4) / 5) % 2 == 0 ? double_buffer.d_keys[double_buffer.selector] : double_buffer.d_keys[double_buffer.selector ^ 1];
//	printf("\n\n\n\nkeys values after sort\n");
//	printDeviceArrays(d_keys, double_buffer.d_values[double_buffer.selector], 50);	


	ContextPtr contextPtr = CreateCudaDevice(0, NULL, false);
	int *d_bucketLabels, *d_bucketValues;	
	cudaMalloc((void**) &d_bucketLabels, BUCKETS * 2 * sizeof(int));
	d_bucketValues = d_bucketLabels + BUCKETS;

	int numSegments;
	simpleReduceByKey<KeyType, ValueType>(*contextPtr, keysPtr, valuesPtr, num_elements, BUCKETS, &numSegments, d_bucketLabels, d_bucketValues);

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
	int *h_buckets = (int*) malloc(BUCKETS * sizeof(int) * 2);
	cudaMemcpy(h_buckets, d_bucketLabels, sizeof(int) * BUCKETS * 2, cudaMemcpyDeviceToHost);

	for (int i = 0; i < BUCKETS; i++)
		h_result[i] = 0;
	for (int i = 0; i < numSegments; i++)
		h_result[h_buckets[i]] = h_buckets[i + BUCKETS];
	free(h_buckets);
	cudaFree(d_bucketLabels);
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
//	printf("rand max is %i\n", RAND_MAX);
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
//		h_keys[i] = index;
                h_keys[i] = rand() % BUCKETS;
                h_values[i] = rand() % 10;
// 		if (i < 50)
//		printf("original key value %i %i %i\n", i, h_keys[i], h_values[i]);
        }
	std::pair<timeval, timeval> result;
	result.first.tv_sec = 0;
	result.second.tv_sec = 0;
        result.first.tv_usec = 0;
        result.second.tv_usec = 0;

	for (int i = 0; i < NO_RUNS; i++)
		result = add(result, sortMR(h_keys, h_values, h_result, num_elements));
	
	cpuMR(h_keys, h_values, result_cpu, num_elements);


	bool correct = true;
	for (int i = 0; i < BUCKETS; i++)
	{
		if (h_result[i] != result_cpu[i])
		{
//			printf("i sense a discrepancy! %i %i %i\n", i, h_result[i], result_cpu[i]);
			correct = false;
		}
	}
	if (correct)
	printf("correct result!\n");
	else
	printf("INCORRECT!!!\n");
	printf("last error is %i\n", cudaGetLastError());

	float time1 = result.first.tv_sec * 1e9 + result.first.tv_usec * 1e3;
	float time2 = result.second.tv_sec * 1e9 + result.second.tv_usec * 1e3;
	printf("%i bins, %i\t%f\t%f\t%f\n", BUCKETS, num_elements, time1 / (num_elements * NO_RUNS), time2 / (num_elements * NO_RUNS), (time1 + time2) / (num_elements * NO_RUNS));

	delete(result_cpu);
	delete(h_keys);
        delete(h_values);
	delete(h_result);
	return 0;
}
