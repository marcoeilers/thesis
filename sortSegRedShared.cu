#include <sys/time.h>
#include "cub/block/block_radix_sort.cuh"
#include "kernels/csrtools.cuh"
#include "kernels/segreducecsr.cuh"


#define BIN_BITS 12
#define BITS 0
//#define START 0
#define START (BIN_BITS - BITS)
#define BUCKETS (1 << BIN_BITS)
#define ITEMS_PER_THREAD 4

#define THREADS_PER_BLOCK 256
#define WORK_PER_BLOCK (ITEMS_PER_THREAD * THREADS_PER_BLOCK)

using namespace mgpu;



struct IndexValue {
	int index;
	int value;
};

struct LabelIndexValue {
	int index;
	int label;
	int value;
};

__global__ void blockSegRed(int *values, int *labels, int *allBuckets, int hists)
{
	int tid = threadIdx.x;
	int *ourValues = values + blockIdx.x * WORK_PER_BLOCK;
	labels = labels + blockIdx.x * WORK_PER_BLOCK;
	int *myResult = allBuckets + blockIdx.x;

	typedef mgpu::CTASegScan<THREADS_PER_BLOCK, mgpu::plus<int> > SegScan;
	typedef cub::BlockRadixSort<int, THREADS_PER_BLOCK, ITEMS_PER_THREAD, IndexValue> BlockRadixSortT;
//	typedef cub::BlockRadixSort<int, THREADS_PER_BLOCK, ITEMS_PER_THREAD, IndexValue, 4, 1, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockRadixSortT;

	union Shared {
		typename BlockRadixSortT::TempStorage sort;
		LabelIndexValue ilvs[THREADS_PER_BLOCK * ITEMS_PER_THREAD];
		struct {
			typename SegScan::Storage segScanStorage;
			int lastValue;
			int lastLabel;
		} segScan;
	};
	__shared__ Shared shared;

	int myLabels[ITEMS_PER_THREAD];

	IndexValue myValues[ITEMS_PER_THREAD];

	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		myLabels[i] = labels[ITEMS_PER_THREAD * tid + i];
		myValues[i].index = ITEMS_PER_THREAD * tid + i;
		myValues[i].value = ourValues[ITEMS_PER_THREAD * tid + i];
	}
	__syncthreads();
	BlockRadixSortT(shared.sort).Sort(myLabels, myValues, 0, BIN_BITS);

	__syncthreads();

	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		shared.ilvs[tid * ITEMS_PER_THREAD + i].index = myValues[i].index;
		shared.ilvs[tid * ITEMS_PER_THREAD + i].label = myLabels[i];
		shared.ilvs[tid * ITEMS_PER_THREAD + i].value = myValues[i].value;
	}
	__syncthreads();

	int myLabelsPred[ITEMS_PER_THREAD];
	int myFlags[ITEMS_PER_THREAD];
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		myValues[i].index = shared.ilvs[tid + i * THREADS_PER_BLOCK].index;
		myValues[i].value = shared.ilvs[tid + i * THREADS_PER_BLOCK].value;
		myLabels[i] = shared.ilvs[tid + i * THREADS_PER_BLOCK].label;
		myLabelsPred[i] = i ? shared.ilvs[tid - 1 + i * THREADS_PER_BLOCK].label : tid ? shared.ilvs[tid - 1 + i * THREADS_PER_BLOCK].label : 0;

		if (i)
			myFlags[i] = shared.ilvs[tid + i * THREADS_PER_BLOCK - 1].label != shared.ilvs[tid + i * THREADS_PER_BLOCK].label;
		else
			myFlags[i] = tid ? shared.ilvs[tid + i * THREADS_PER_BLOCK - 1].label != shared.ilvs[tid + i * THREADS_PER_BLOCK].label : 1;
	}

	__syncthreads();
//                        atomicAdd(&myResult[curInd * hists], curVal);

	int carryOut;
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		int x = SegScan::SegScan(tid, myValues[i].value, myFlags[i], shared.segScan.segScanStorage, &carryOut, 0);

		__syncthreads();

		if (myFlags[i] && myLabels[i] != 0)
		{
//			allBuckets[((myLabelsPred[i]) * pitchByInt) + blockIdx.x] = (i && myLabelsPred[i] == shared.segScan.lastLabel) ? x + shared.segScan.lastValue : x;
			myResult[myLabelsPred[i] * hists] = (i && myLabelsPred[i] == shared.segScan.lastLabel) ? x + shared.segScan.lastValue : x;
		}


		__syncthreads();
		if (threadIdx.x == (THREADS_PER_BLOCK - 1))
		{
			myResult[myLabels[ITEMS_PER_THREAD - 1] * hists] = carryOut;
//                        allBuckets[((myLabels[ITEMS_PER_THREAD - 1]) * pitchByInt) + blockIdx.x] = carryOut;
			shared.segScan.lastValue = carryOut;
			shared.segScan.lastLabel = myLabels[i];
		}
	}

}

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
std::pair<timeval, timeval> sortMR(int *h_keys, int *h_values, int *h_result, int num_elements, CudaContext &context)
{
	typedef int KeyType;
	typedef int ValueType;
	
	// Allocate device data. (We will let the sorting enactor create
	// the "pong" storage if/when necessary.)
	KeyType *d_keys;
	ValueType *d_values;
	int *d_result, *d_allBuckets;
	cudaMalloc((void**) &d_keys, sizeof(KeyType) * num_elements);
	cudaMalloc((void**) &d_values, sizeof(ValueType) * num_elements);

	// Copy host data to device data
	cudaMemcpy(d_keys, h_keys, sizeof(KeyType) * num_elements, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, sizeof(ValueType) * num_elements, cudaMemcpyHostToDevice);

	timeval before, between, after;
	
	gettimeofday(&before, NULL);

	cudaThreadSynchronize();

        gettimeofday(&between, NULL);

	int blocks = num_elements / (THREADS_PER_BLOCK * ITEMS_PER_THREAD);
	int threadspb = THREADS_PER_BLOCK;
	int hists = blocks;

	
        cudaMalloc((void**) &d_allBuckets, sizeof(int) * BUCKETS * hists);

	cudaMemset(d_allBuckets, 0, sizeof(int) * BUCKETS * hists);

//__global__ void blockSegRed(int *values, int *labels, int *result, int *allBuckets, size_t pitch)
	blockSegRed<<<blocks, threadspb>>>(d_values, d_keys, d_allBuckets, hists);
//        multiReduceCombine(double_buffer.d_keys[double_buffer.selector], double_buffer.d_values[double_buffer.selector], d_allBuckets, num_elements, blocks, threadspb, BUCKETS, hists);

        cudaThreadSynchronize();


        gettimeofday(&after, NULL);


	step_iterator<int> segmentStarts(0, hists);

	cudaMalloc((void**) &d_result, BUCKETS * sizeof(int));

	SegReduceCsr(d_allBuckets, segmentStarts, BUCKETS * hists, BUCKETS, false, d_result, (int)0, mgpu::plus<int>(), context);

	cudaMemcpy(h_result, d_result, sizeof(int) * BUCKETS, cudaMemcpyDeviceToHost);

	cudaFree(d_allBuckets);
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
        ContextPtr contextPtr = mgpu::CreateCudaDevice(0, NULL, false);

	printf("rand max is %i\n", RAND_MAX);
	typedef int KeyType;
	typedef int ValueType;

	unsigned int num_elements = (THREADS_PER_BLOCK * ITEMS_PER_THREAD * 8) * 8192;


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
	result = sortMR<START, BITS>(h_keys, h_values, h_result, num_elements, *contextPtr);
	
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
