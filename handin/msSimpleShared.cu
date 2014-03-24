#include <stdio.h>
#include <sys/time.h>
#include "cudpp.h"

#define BINS_EXP 9
#define BINS (1 << BINS_EXP)
#define BUCKETS BINS
#define NT 32
#define VT 8
#define VT1 (VT + 1)



void cudppMS(int blocks, int *in, int *out, size_t pitch);


__global__ void multiReduceKernel(int *indices, int *values, int *result, int *allBuckets, int segLength, size_t pitch)
{
__shared__ int blockBuckets[NT * BUCKETS];
__shared__ int sIndices[VT1 * NT];
__shared__ int sValues[VT1 * NT];

        int *myBuckets = &blockBuckets[threadIdx.x];

        int blocks = gridDim.x;
        int blockId = blockIdx.x;
        int threadId = threadIdx.x;
//        int threadspb = blockDim.x;
        int hists = blocks * NT;
        int todopb = NT * segLength;

        int *ourIndices = indices + blockId * todopb;
        int *ourValues = values + blockId * todopb;
        int *ourResult = result + blockId * todopb;
        int *myResult = ourResult + threadId * segLength;
        int *myIndices = &sIndices[threadId * VT1];
        int *myValues = &sValues[threadId * VT1];

        // initialize buckets to zero
        for (int i = threadIdx.x; i < NT * BUCKETS; i += NT)
                blockBuckets[i] = 0;

        __syncthreads();

        #pragma unroll 16
        for (int it = 0; it < segLength; it += VT)
        {
        int offset = it;
        int todo = min(segLength - offset, VT);

        for(int i = threadIdx.x; i < todo * NT; i += NT)
        {
                int segm = i / todo;
                int ind = i % todo;
                int sIndex = segm * VT1 + ind;
                int index = segm * segLength + ind + offset;
                sIndices[sIndex] = ourIndices[index];
                sValues[sIndex] = ourValues[index];
        }

        __syncthreads();

        for (int i = 0; i < todo; i++)
        {
                int currentLabel = myIndices[i];
                myResult[offset + i] = myBuckets[currentLabel * NT];
                myBuckets[currentLabel * NT] += myValues[i];
        }

        __syncthreads();

        }

        int *myGlobalBuckets = allBuckets + blockIdx.x * NT + threadId;
	int pitchByInt = pitch / sizeof(int);
        // write results to global memory
        for (int i = 0; i < BUCKETS; i++)
        {
//if (blockIdx.x == 0 && threadIdx.x < 4) printf("thread %i writing result %i for bucket %i\n", threadIdx.x, blockBuckets[threadId + i * NT], i);
                myGlobalBuckets[i * pitchByInt] = blockBuckets[threadId + i * NT];
//                myResult[i * hists] = blockBuckets[threadId + i * NT];
        }
}


/*
__global__ void blockSegScan(int *values, int *labels, int *result, int *allBuckets, size_t pitch)
{
	int tid = threadIdx.x;
	int *ourValues = values + blockIdx.x * WORK_PER_BLOCK;
	labels = labels + blockIdx.x * WORK_PER_BLOCK;
	result = result + blockIdx.x * WORK_PER_BLOCK;
	int pitchByInt = pitch / sizeof(int);
//	for (int i = tid; i < BINS; i += THREADS_PER_BLOCK)
//		allBuckets[i * pitchByInt + blockIdx.x] = 0;

	
//        for (int i = tid; i < pitchByInt * BINS; i += THREADS_PER_BLOCK)
//                allBuckets[blockIdx.x * BINSi] = 0;
//        __syncthreads();

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
	BlockRadixSortT(shared.sort).Sort(myLabels, myValues, 0, BINS_EXP);

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

	int carryOut;
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		int x = SegScan::SegScan(tid, myValues[i].value, myFlags[i], shared.segScan.segScanStorage, &carryOut, 0);

		__syncthreads();

		if (myFlags[i] && myLabels[i] != 0)
		{
			allBuckets[((myLabelsPred[i]) * pitchByInt) + blockIdx.x] = (i && myLabelsPred[i] == shared.segScan.lastLabel) ? x + shared.segScan.lastValue : x;
		}

                if (i != 0 && myLabels[i] == shared.segScan.lastLabel)
                {
                        x += shared.segScan.lastValue;
                }

                int writeResult = myFlags[i] ? 0 : x;

		result[myValues[i].index] = writeResult;


		__syncthreads();
		if (threadIdx.x == (THREADS_PER_BLOCK - 1))
		{
                        allBuckets[((myLabels[ITEMS_PER_THREAD - 1]) * pitchByInt) + blockIdx.x] = carryOut;
			shared.segScan.lastValue = carryOut;
			shared.segScan.lastLabel = myLabels[i];
		}
	}

}
*/


/*
__global__
void addRemainder(int *bucketSums, int *indices, int *result, size_t pitch)
{
__shared__ int bins[BINS];
int pitchByInt = pitch / sizeof(int);
int work = WORK_PER_BLOCK;

int blocks = gridDim.x;

#pragma unroll 2
for (int i = threadIdx.x; i < BINS; i+=THREADS_PER_BLOCK)
{
	bins[i] = bucketSums[blockIdx.x + i * pitchByInt];
}
__syncthreads();


int *myIndices = &indices[work * blockIdx.x];
int *myResult = &result[work * blockIdx.x];

#pragma unroll ITEMS_PER_THREAD
for (int i = threadIdx.x; i < work; i+=THREADS_PER_BLOCK)
{
//if (blockIdx.x == 1 && i < 100) printf("adding %i to result %i, label is %i\n", bins[myIndices[i]], i + blockIdx.x * work, myIndices[i]);
	myResult[i] += bins[myIndices[i]];
}

}
*/

__global__
void addRemainderGlobalShared(int *bucketSums, int *indices, int *result, int segLength, size_t pitch)
{
int pitchByInt = pitch / sizeof(int);

int blocks = gridDim.x;
int nThreads = blockDim.x;
__syncthreads();

int *myIndices = &indices[segLength * blockIdx.x];
int *myResult = &result[segLength * blockIdx.x];

#pragma unroll ITEMS_PER_THREAD
for (int i = threadIdx.x; i < segLength; i+= nThreads)
{
        myResult[i] += bucketSums[blockIdx.x + (myIndices[i]) * pitchByInt]; //bins[myIndices[i]];
}

}

/*
void comparePartials(int *d_indices, int *d_values, int *d_results, int *d_allBuckets, int num_elements, size_t pitch)
{
int pitchByInt = pitch / sizeof(int);
int *indices, *values, *results, *allBuckets, *cresults, *callBuckets;
indices = (int*) malloc(num_elements * sizeof(int) * 4);
values = indices + num_elements;
results = values + num_elements;
cresults = results + num_elements;
allBuckets = (int*) malloc(pitch * BINS * 2);
callBuckets = allBuckets + pitchByInt * BINS;

cudaMemcpy(indices, d_indices, sizeof(int) * num_elements, cudaMemcpyDeviceToHost);
cudaMemcpy(values, d_values, sizeof(int) * num_elements, cudaMemcpyDeviceToHost);
cudaMemcpy(results, d_results, sizeof(int) * num_elements, cudaMemcpyDeviceToHost);
cudaMemcpy(allBuckets, d_allBuckets, pitch * BINS, cudaMemcpyDeviceToHost);

for (int i = 0; i < num_elements; i += NT * VT)
{
	int block = i / (THREADS_PER_BLOCK * ITEMS_PER_THREAD);

	int *curIndices = indices + i;
	int *curValues = values + i;
	int *curResults = cresults + i;

	for (int j = 0; j < BINS; j++)
		callBuckets[j] = 0;
	for (int j = 0; j < THREADS_PER_BLOCK * ITEMS_PER_THREAD; j++)
	{
		curResults[j] = callBuckets[curIndices[j]];
		callBuckets[curIndices[j]] += curValues[j];
	}

	for (int j = 0; j < THREADS_PER_BLOCK * ITEMS_PER_THREAD; j++)
	{
		if (results[i + j] != curResults[j])
			printf("different result block %i in %i should be %i is %i, label %i\n", block, i + j, curResults[j], results[i+j], curIndices[j]);
	}

	for (int j = 0; j < BINS; j++)
	{
		if (callBuckets[j] != allBuckets[j * pitchByInt + block])
			printf("different bucket block %i bin %i is %i should be %i\n", block, j, allBuckets[j * pitchByInt + block], callBuckets[j]);
	}
}

}
*/


void invokeMultiScan(int *indices, int *values, int *results, int num_elements)
{
cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);

int shmemPerBlock = NT * BUCKETS * sizeof(int) + (2 * NT * VT1 * sizeof(int));
int blocksPerSm = props.sharedMemPerBlock / shmemPerBlock;

int totalBlocks = props.multiProcessorCount * blocksPerSm;
printf("%i SMs, %i bytes shared memory, we use %i bytes per block, therefore have %i blocks and segLEngth %i\n", props.multiProcessorCount, props.sharedMemPerBlock, shmemPerBlock, totalBlocks, num_elements / (NT * totalBlocks));

int hists = NT * totalBlocks;


int segLength = num_elements / hists;

int *allBuckets;
int *allBucketsResult;
size_t pitch;
size_t pitch_result;
cudaMallocPitch((void**)&allBuckets, &pitch, hists * sizeof(int),  BINS);
cudaMallocPitch((void**)&allBucketsResult, &pitch_result, hists * sizeof(int), BINS);

printf("temp memory allocation error: %i\n", cudaGetLastError());

dim3 dimBlock(NT, 1);
dim3 dimGrid(totalBlocks, 1);
cudaFuncSetCacheConfig(multiReduceKernel, cudaFuncCachePreferShared);
cudaFuncSetCacheConfig(addRemainderGlobalShared, cudaFuncCachePreferShared);


timeval before, between1, between2, after;

gettimeofday(&before, NULL);
//cudaMemset(results, 0, num_elements * sizeof(int));
cudaMemset(allBuckets, 0, pitch * BINS);

//naiveGlobalSeparateMultiScan<<<blocks, THREADS_PER_BLOCK>>>(indices, values, allBuckets, pitch);
multiReduceKernel<<<totalBlocks, NT>>>(indices, values, results, allBuckets, segLength, pitch);

cudaThreadSynchronize();
printf("error after blockSegScan is %i\n", cudaGetLastError());
//comparePartials(indices, values, results, allBuckets, num_elements, pitch);

gettimeofday(&between1, NULL);

cudppMS(hists, allBuckets, allBucketsResult, pitch);
cudaThreadSynchronize();

gettimeofday(&between2, NULL);

	addRemainderGlobalShared<<<hists, 128>>>(allBucketsResult, indices, results, segLength, pitch);


cudaThreadSynchronize();
gettimeofday(&after, NULL);

float time3 = (after.tv_sec - between2.tv_sec) * 1e9 + (after.tv_usec - between2.tv_usec) * 1e3;
float time2 = (between2.tv_sec - between1.tv_sec) * 1e9 + (between2.tv_usec - between1.tv_usec) * 1e3;
float time1 = (between1.tv_sec - before.tv_sec) * 1e9 + (between1.tv_usec - before.tv_usec) * 1e3;
printf("%i\t%f\t%f\t%f\t%f\n", BUCKETS, time1 / num_elements, time2 / num_elements, time3 / num_elements, (time1 + time2 + time3) / num_elements);
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
//if (i == 3 * THREADS_PER_BLOCK * ITEMS_PER_THREAD) {for (int j = 0; j < BINS; j++) printf("in block 3 bucket %i should be %i\n", j, buckets[j]);}

	result[i] = buckets[indices[i]];
	buckets[indices[i]] += values[i];
}
}



int main()
{
cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);

int shmemPerBlock = NT * BUCKETS * sizeof(int) + (2 * NT * VT1 * sizeof(int));
int blocksPerSm = props.sharedMemPerBlock / shmemPerBlock;

int totalBlocks = props.multiProcessorCount * blocksPerSm;
//printf("%i SMs, %i bytes shared memory, we use %i bytes per block, therefore have %i blocks\n", props.multiProcessorCount, props.sharedMemPerBlock, shmemPerBlock, totalBlocks);

int hists = NT * totalBlocks;


int size = ((1 << 26) / hists) * hists;
int *keys = (int*) malloc(size * sizeof(int));
int *values = (int*) malloc(size * sizeof(int));
int *result = (int*) malloc(size * sizeof(int));
int *result_cpu = (int*) malloc(size * sizeof(int));
int index = rand() % BINS; 
int *d_keys, *d_values, *d_results;

{
for (int i = 0; i < size; i++)
{
	values[i] = rand() % 10;
	keys[i] = rand() % BUCKETS;
//if (i < 4 * ITEMS_PER_THREAD)
//	printf("original index %i label %i value %i \n", i, keys[i], values[i]);
	result_cpu[i] = 15;
}


cudaMalloc((void**) &d_keys, size * sizeof(int));
cudaMalloc((void**) &d_values, size * sizeof(int));
cudaMalloc((void**) &d_results, size * sizeof(int));

cudaMemcpy(d_keys, keys, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, size * sizeof(int), cudaMemcpyHostToDevice);

invokeMultiScan(d_keys, d_values, d_results, size);

cudaMemcpy(result, d_results, size * sizeof(int), cudaMemcpyDeviceToHost);

timeval before, after;
gettimeofday(&before, NULL);
multiScanCpu(keys, values, result_cpu, size);
gettimeofday(&after, NULL);

printf("cpu needed %f ns/int \n", ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / size);
int correct = 1;
for (int i = 0; i < size; i++)
{
	if (result[i] != result_cpu[i])
	{
//		printf("I sense a discrepancy! %i %i %i value %i label %i\n", i, result[i], result_cpu[i], values[i], keys[i]);
		correct = 0;
	}
//	else
//		printf("correct!\n");
}
printf("correct? %i\n", correct);
printf("last error: %i\n", cudaGetLastError());

cudaFree(d_keys);
cudaFree(d_values);
cudaFree(d_results);
}

{
printf("now deg\n");
for (int i = 0; i < size; i++)
{
        values[i] = rand() % 10;
        keys[i] = index;
//if (i < 4 * ITEMS_PER_THREAD)
//      printf("original index %i label %i value %i \n", i, keys[i], values[i]);
        result_cpu[i] = 15;
}


cudaMalloc((void**) &d_keys, size * sizeof(int));
cudaMalloc((void**) &d_values, size * sizeof(int));
cudaMalloc((void**) &d_results, size * sizeof(int));

cudaMemcpy(d_keys, keys, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, size * sizeof(int), cudaMemcpyHostToDevice);

invokeMultiScan(d_keys, d_values, d_results, size);

cudaMemcpy(result, d_results, size * sizeof(int), cudaMemcpyDeviceToHost);

timeval before, after;
gettimeofday(&before, NULL);
multiScanCpu(keys, values, result_cpu, size);
gettimeofday(&after, NULL);

printf("cpu needed %f ns/int \n", ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / size);
int correct = 1;
for (int i = 0; i < size; i++)
{
        if (result[i] != result_cpu[i])
        {
//              printf("I sense a discrepancy! %i %i %i value %i label %i\n", i, result[i], result_cpu[i], values[i], keys[i]);
                correct = 0;
        }
//      else
//              printf("correct!\n");
}
printf("correct? %i\n", correct);
printf("last error: %i\n", cudaGetLastError());

cudaFree(d_keys);
cudaFree(d_values);
cudaFree(d_results);

}
}


