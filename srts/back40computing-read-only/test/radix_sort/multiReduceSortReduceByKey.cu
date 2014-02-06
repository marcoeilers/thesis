#include <part_sort.cuh>
#include <sys/time.h>
#include "kernels/reducebykey.cuh"

using namespace mgpu;

#define PREPROCESS true
#define BIN_BITS 16           
#define EXP 26
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

template<typename KeyType, typename ValType>
void simpleReduceByKey(CudaContext& context, KeyType *indices, ValType *values, ValType *result, int num_elements, int buckets, bool preprocess)
{
//	MGPU_MEM(int) countsDevice = context.Malloc<int>(1);

	std::auto_ptr<ReduceByKeyPreprocessData> preprocessData;
	int numSegments;
//	if(preprocess) {
		ReduceByKeyPreprocess<ValType>(num_elements, indices, 
			(KeyType*)0, mgpu::equal_to<KeyType>(),
			&numSegments, (int*)0, &preprocessData, context);
//	}

//	context.Start();
	
//	if(preprocess) {
		ReduceByKeyApply(*preprocessData, values, (ValType)0, 
			mgpu::plus<ValType>(), result, context);
/*	} else {
		ReduceByKey(indices, values, num_elements, (ValType)0,
			mgpu::plus<ValType>(), mgpu::equal_to<ValType>(),
			(KeyType*)0, result,
			(int*)0, 
			countsDevice->get(), context);
	}
*/
//	if(!preprocess)
//		copyDtoH(buckets, countsDevice->get(), 1);
}


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
//	enactor.Sort(double_buffer, num_elements);
	enactor.OneRunSort<0, BIN_BITS>(double_buffer, num_elements);


	cudaThreadSynchronize();

        gettimeofday(&between, NULL);

//	printf("\n\n\n\nkeys values after sort\n");
//	printDeviceArrays(d_keys, double_buffer.d_values[double_buffer.selector], 50);	

        cudaMalloc((void**) &d_result, sizeof(int) * BUCKETS);
	ContextPtr contextPtr = CreateCudaDevice(0, NULL, false);
	
	simpleReduceByKey<KeyType, ValueType>(*contextPtr, d_keys, double_buffer.d_values[double_buffer.selector], d_result, num_elements, BUCKETS, PREPROCESS);

        cudaThreadSynchronize();
        gettimeofday(&after, NULL);

	// Cleanup "pong" storage
	if (double_buffer.d_keys[double_buffer.selector ^ 1]) {
		cudaFree(double_buffer.d_keys[double_buffer.selector ^ 1]);
	}
	if (double_buffer.d_values[double_buffer.selector ^ 1]) {
		cudaFree(double_buffer.d_values[double_buffer.selector ^ 1]);
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

        for (int i = 0; i < num_elements; ++i)
        {
                h_keys[i] = rand() % BUCKETS;
                h_values[i] = rand() % 100;
// 		if (i < 50)
//		printf("original key value %i %i %i\n", i, h_keys[i], h_values[i]);
        }
	std::pair<timeval, timeval> result;
	result = sortMR(h_keys, h_values, h_result, num_elements);
	
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
	printf("%i\t%f\t%f\t%f\n", num_elements, time1 / num_elements, time2 / num_elements, (time1 + time2) / num_elements);

	delete(result_cpu);
	delete(h_keys);
        delete(h_values);
	delete(h_result);
	return 0;
}
