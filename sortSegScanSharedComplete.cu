#include <stdio.h>
#include <sys/time.h>
#include "cudpp.h"
#include "cub/block/block_radix_sort.cuh"
#include "kernels/csrtools.cuh"

#define ITEMS_PER_THREAD 3
#define BINS 64
#define THREADS_PER_BLOCK 64

//#define SHEFFLER
#define WORK_PER_BLOCK 32768




//TODO bucket results are stored in the wrong order
// which doesnt influence performance i think but makes results wrong


void cudppMS(int blocks, int *in, int *out, size_t pitch);

struct IndexValue {
	int index;
	int value;
};

struct LabelIndexValue {
	int index;
	int label;
	int value;
};



__global__ void blockSegScan(int *values, int *labels, int *result, int *allBuckets, size_t pitch)
{
	int tid = threadIdx.x;
	values = values + blockIdx.x * WORK_PER_BLOCK;
	labels = labels + blockIdx.x * WORK_PER_BLOCK;
	result = result + blockIdx.x * WORK_PER_BLOCK;
	int pitchByInt = pitch / sizeof(int);

	typedef mgpu::CTASegScan<THREADS_PER_BLOCK, mgpu::plus<int> > SegScan;
	typedef cub::BlockRadixSort<int, THREADS_PER_BLOCK, ITEMS_PER_THREAD, IndexValue> BlockRadixSortT;

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
//printf("size of shared %i, sort %i, labels %i, segScanStorage %i\n", sizeof(shared), sizeof(shared.sort), sizeof(shared.labels), sizeof(shared.segScanStorage));
	int myLabels[ITEMS_PER_THREAD];
	IndexValue myValues[ITEMS_PER_THREAD];
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		myLabels[i] = labels[ITEMS_PER_THREAD * tid + i];
		myValues[i].index = ITEMS_PER_THREAD * tid + i;
		myValues[i].value = values[ITEMS_PER_THREAD * tid + i];
	}

	BlockRadixSortT(shared.sort).Sort(myLabels, myValues);

	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		shared.ilvs[tid * ITEMS_PER_THREAD + i].index = myValues[i].index;
		shared.ilvs[tid * ITEMS_PER_THREAD + i].label = myLabels[i];
		shared.ilvs[tid * ITEMS_PER_THREAD + i].value = myValues[i].value;
	}
	__syncthreads();

	int myFlags[ITEMS_PER_THREAD];
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		myValues[i].index = shared.ilvs[tid + i * THREADS_PER_BLOCK].index;
		myValues[i].value = shared.ilvs[tid + i * THREADS_PER_BLOCK].value;
		myLabels[i] = shared.ilvs[tid + i * THREADS_PER_BLOCK].label;
		if (i)
			myFlags[i] = shared.ilvs[tid + i * THREADS_PER_BLOCK - 1].label != shared.ilvs[tid + i * THREADS_PER_BLOCK].label;
		else
			myFlags[i] = tid ? shared.ilvs[tid + i * THREADS_PER_BLOCK - 1].label != shared.ilvs[tid + i * THREADS_PER_BLOCK].label : 1;
	}

	__syncthreads();
	int carryOut;
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		int x = SegScan::SegScan(tid, myValues[i].value, myFlags[i], shared.segScan.segScanStorage, &carryOut, 0);

		if (i != 0 && myLabels[i] == shared.segScan.lastLabel)
		{
			x += shared.segScan.lastValue;
		}


		int writeResult = myFlags[i] ? 0 : x;
		if (myFlags[i])
		{
			allBuckets[((myLabels[i] - 1) * pitchByInt) + blockIdx.x] = x;
		}

		result[myValues[i].index] = writeResult;
		
		__syncthreads();
		if (threadIdx.x == (THREADS_PER_BLOCK - 1))
		{
			shared.segScan.lastValue = x + myValues[i].value;
			shared.segScan.lastLabel = myLabels[i];
		}
	}
	if (threadIdx.x == (THREADS_PER_BLOCK - 1))
		allBuckets[((myLabels[ITEMS_PER_THREAD - 1] - 1) * pitchByInt) + blockIdx.x] = carryOut;
}




__global__
void addRemainder(int *bucketSums, int *indices, int *result)
{
__shared__ int bins[BINS];

int work = WORK_PER_BLOCK;

int blocks = gridDim.x;

#pragma unroll
for (int i = threadIdx.x; i < BINS; i+=32)
{
	bins[i] = bucketSums[blockIdx.x + i * blocks];
}
__syncthreads();


int *myIndices = &indices[work * blockIdx.x];
int *myResult = &result[work * blockIdx.x];
#pragma unroll
for (int i = threadIdx.x; i < work; i+=32)
{
	myResult[i] += bins[myIndices[i]];
}

}



void invokeMultiScan(int *indices, int *values, int num_elements)
{
int blocks = num_elements / (ITEMS_PER_THREAD * THREADS_PER_BLOCK);

int *allBuckets;
int *allBucketsResult;
size_t pitch;
size_t pitch_result;
cudaMallocPitch((void**)&allBuckets, &pitch, blocks * sizeof(int),  BINS);
cudaMallocPitch((void**)&allBucketsResult, &pitch_result, blocks * sizeof(int), BINS);

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);
cudaFuncSetCacheConfig(allMultiScan, cudaFuncCachePreferShared);
cudaFuncSetCacheConfig(addRemainder, cudaFuncCachePreferShared);


timeval before, between1, between2, after;

gettimeofday(&before, NULL);

//naiveGlobalSeparateMultiScan<<<blocks, THREADS_PER_BLOCK>>>(indices, values, allBuckets, pitch);
blockSegScan<<<blocks, THREADS_PER_BLOCK>>>(values, indices, values, allBuckets, pitch);


cudaThreadSynchronize();


gettimeofday(&between1, NULL);

cudppMS(blocks, allBuckets, allBucketsResult, pitch);
cudaThreadSynchronize();

gettimeofday(&between2, NULL);

addRemainder<<<dimGrid, dimBlock>>>(allBucketsResult, indices, values);


cudaThreadSynchronize();
gettimeofday(&after, NULL);

float time3 = (after.tv_sec - between2.tv_sec) * 1e9 + (after.tv_usec - between2.tv_usec) * 1e3;
float time2 = (between2.tv_sec - between1.tv_sec) * 1e9 + (between2.tv_usec - between1.tv_usec) * 1e3;
float time1 = (between1.tv_sec - before.tv_sec) * 1e9 + (between1.tv_usec - before.tv_usec) * 1e3;
printf("%i\t%f\t%f\t%f\t%f\n", num_elements, time1 / num_elements, time2 / num_elements, time3 / num_elements, (time1 + time2 + time3) / num_elements);
}

void cudppMS(int blocks, int *in, int *out, size_t pitch)
{
// Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, blocks, BINS, pitch / sizeof(int));  
    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

// Run the scan
    res = cudppMultiScan(scanplan, out, in, blocks, BINS);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }
}





void multiScanCpu(int *indices, int *values, int *result, int num_elements)
{
int buckets[BINS];

for (int i = 0; i < BINS; i++)
	buckets[i] = 0;

for (int i = 0; i < num_elements; i++)
{
	result[i] = buckets[indices[i]];
	buckets[indices[i]] += values[i];
}
}






int main()
{
printf("spinerec hsa size %i\n", sizeof(spinerec));
int size = (8 * 3 * 1024) * 10900;
int *keys = (int*) malloc(size * sizeof(int));
int *values = (int*) malloc(size * sizeof(int));
int *result = (int*) malloc(size * sizeof(int));
int *result_cpu = (int*) malloc(size * sizeof(int));
int index = rand() % BINS; 
for (int i = 0; i < size; i++)
{
	values[i] = index;
	keys[i] = rand() % BINS;
//if (i < 5)
//	printf("original value %i %i \n", i, values[i]);
//	result[i] = 0;
}

int *d_keys, *d_values;
cudaMalloc((void**) &d_keys, size * sizeof(int));
cudaMalloc((void**) &d_values, size * sizeof(int));

cudaMemcpy(d_keys, keys, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, size * sizeof(int), cudaMemcpyHostToDevice);

invokeMultiScan(d_keys, d_values, size);

cudaMemcpy(result, d_values, size * sizeof(int), cudaMemcpyDeviceToHost);

timeval before, after;
gettimeofday(&before, NULL);
multiScanCpu(keys, values, result_cpu, size);
gettimeofday(&after, NULL);

printf("cpu needed %f ns/int \n", ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / size);

for (int i = 0; i < size; i++)
{
	if (result[i] != result_cpu[i])
		printf("I sense a discrepancy! %i %i %i value %i\n", i, result[i], result_cpu[i], values[i]);
//	else
//		printf("correct!\n");
}

printf("last error: %i\n", cudaGetLastError());
}


