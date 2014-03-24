#include <b40c/radix_sort/enactor.cuh>
#include <b40c/util/multi_buffer.cuh>

#include <sys/time.h>
#include <scatterKernel.cuh>
#include <assert.h>
#include <thrust/scan.h>
#include "valIndex.h"


#define CHECK 128
#define BINS_EXP 26
#define ELEMENTS_EXP 25

__global__ void toValIndex(int *in, valIndex *out, int num_elements){
int todo = num_elements / gridDim.x;
int offset = todo * blockIdx.x;

for (int i = threadIdx.x; i < todo; i += blockDim.x){
int curOffset = offset + i;
int curVal = in[curOffset];
valIndex temp;
temp.index = curOffset;
temp.value = curVal;
out[curOffset] = temp;

}
}

__global__ void toInt(valIndex *in, int *out, int *indices, int num_elements){
int todo = num_elements / gridDim.x;
int offset = todo * blockIdx.x;

for (int i = threadIdx.x; i < todo; i += blockDim.x){
int curOffset = offset + i;
valIndex cur = in[curOffset];
out[curOffset] = cur.value;
indices[curOffset] = cur.index;
}
}


void multiScanCpu(int *keys, int *values, int *result, int num_elements)
{
	int *buckets = (int*) malloc((1 << BINS_EXP) *sizeof(int));
	for (int i = 0; i < (1 << BINS_EXP); i++)
		buckets[i] = 0;

	for (int i = 0; i < num_elements; i++)
	{
		buckets[keys[i]] += values[i];
		result[i] = buckets[keys[i]];
//                buckets[keys[i]] += values[i].value;
	}
	free(buckets);
}  

template<typename ValueType, int START_BIT, int NO_BITS>
std::pair<timeval, timeval> multiScan(int *h_keys, ValueType *h_values, int *h_result, int num_elements, bool keys_only)
{
	typedef int KeyType;
	
	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	ValueType *d_values;
	valIndex *d_valIndices;
	int *d_result;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements);
	cudaMalloc((void**) &d_valIndices, sizeof(valIndex) * num_elements);
	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice);

	timeval before, between, after;
	
	gettimeofday(&before, NULL);

	toValIndex<<<8 * 16, 128>>>(d_values, d_valIndices, num_elements);	
	cudaThreadSynchronize();
	// Create a reusable sorting enactor
	b40c::radix_sort::Enactor enactor;

	// Create ping-pong storage wrapper
	b40c::util::DoubleBuffer<KeyType, double> double_buffer;

	// The current key buffer (double_buffer.d_keys[double_buffer.selector]) backs the keys.
	double_buffer.d_keys[double_buffer.selector] = d_keys;
	double_buffer.d_values[double_buffer.selector] = (double*)d_valIndices;

	// Allocate pong buffer
        int *d_double_keys;
	double *d_double_values;

        cudaMalloc((void**) &d_double_keys, sizeof(KeyType) * num_elements);
        cudaMalloc((void**) &d_double_values, sizeof(valIndex) * num_elements);
        double_buffer.d_keys[double_buffer.selector ^ 1] = d_double_keys;
        double_buffer.d_values[double_buffer.selector ^ 1] = d_double_values;


	// Sort
//	enactor.Sort(double_buffer, num_elements);
	enactor.OneRunSort<START_BIT, NO_BITS>(double_buffer, num_elements);

	cudaThreadSynchronize();

	gettimeofday(&between, NULL);
	toInt<<<8 * 16, 128>>>((valIndex*)double_buffer.d_values[double_buffer.selector], d_values, double_buffer.d_keys[double_buffer.selector ^ 1], num_elements);
	
	thrust::equal_to<int> binary_pred;
	thrust::plus<int>     binary_op;
	thrust::device_ptr<int> keys_ptr(double_buffer.d_keys[double_buffer.selector]);
	thrust::device_ptr<int> values_ptr(d_values);

	thrust::inclusive_scan_by_key(keys_ptr, keys_ptr + num_elements, values_ptr, values_ptr, binary_pred, binary_op);
	

        cudaMalloc((void**) &d_result, sizeof(int) * num_elements);
	scatter(double_buffer.d_keys[double_buffer.selector ^ 1], d_values, d_result, num_elements);
//        scatterValIndex((valIndex*)double_buffer.d_values[double_buffer.selector], d_result, num_elements);

        cudaThreadSynchronize();
        gettimeofday(&after, NULL);

	// Cleanup "pong" storage
	if (d_double_values) {
		cudaFree(d_double_values);
	}
	if (d_double_keys) {
		cudaFree(d_double_keys);
	}

	cudaMemcpy(h_result, d_result, sizeof(int) * num_elements, cudaMemcpyDeviceToHost);


	cudaFree(d_keys);
	cudaFree(d_values);
	cudaFree(d_valIndices);

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
	typedef int KeyType;
	typedef int ValueType;;
printf("valIndex %i float %i double %i\n", sizeof(valIndex), sizeof(float), sizeof(double));
	unsigned int num_elements = 1 << ELEMENTS_EXP;


        // Allocate host problem data
	KeyType *h_keys = new KeyType[num_elements];
	ValueType *h_values = new ValueType[num_elements];
	int *h_result = new int[num_elements];
	int *h_result_cpu = new int[num_elements];

        // Initialize host problem data
	int index = rand() % (1 << BINS_EXP);
        for (int i = 0; i < num_elements; ++i)
        {
                h_keys[i] = rand() % (1 << BINS_EXP);
//                h_values[i] = i;
		h_values[i] = rand() % 1000;
		h_result[i] = 1;
//		printf("%i\t%i\t%i\n", i, h_keys[i], h_values[i].value);
        }
//printf("\n\n\n");
	timeval before, after;
	gettimeofday(&before, NULL);
	multiScanCpu(h_keys, h_values, h_result_cpu, num_elements);
	gettimeofday(&after, NULL);
	std::pair<timeval, timeval> result;
	result = multiScan<int, 0, BINS_EXP>(h_keys, h_values, h_result, num_elements, false);
	printf("%i buckets cpu gpu ns/int %f\t%f\n", (1 << BINS_EXP), ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements, ((result.first.tv_sec * 1e9 + result.first.tv_usec * 1e3) + (result.second.tv_sec * 1e9 + result.second.tv_usec * 1e3)) / num_elements);
printf("now doing that checking thing...\n");
	for (int i = 0; i < num_elements; i++)
	{
		if(h_result[i] != h_result_cpu[i])
			printf("results unequal: %i %i %i\n", i, h_result[i], h_result_cpu[i]);
	}

	delete(h_keys);
        delete(h_values);
	delete(h_result);
	return 0;
}
