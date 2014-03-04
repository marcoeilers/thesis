#include <stdio.h>
#include <sys/time.h>
#include "cudpp.h"

#define BINS 64
#define THREADS_PER_BLOCK 64

//#define SHEFFLER
#define WORK_PER_BLOCK 32768


#define FETCHLABELS

//TODO bucket results are stored in the wrong order
// which doesnt influence performance i think but makes results wrong

struct spinerec {
	int rowsum;
	int spinesum;
	int spine;
};
void cudppMS(int blocks, int *in, int *out, size_t pitch);


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

__global__
void addRemainder(int *bucketSums, int *indices, int *result)
{
__shared__ int bins[BINS];
#ifdef SHEFFLER
int work = 1024;
#else
int work = WORK_PER_BLOCK;
#endif

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


__global__
void naiveMultiScan(int *labels, int *values, int *allBuckets, size_t pitch)
{
__shared__ int buckets[BINS];
__shared__ int curLabels[THREADS_PER_BLOCK];
__shared__ int curValues[THREADS_PER_BLOCK];

for (int i = threadIdx.x; i < BINS; i += THREADS_PER_BLOCK)
	buckets[i] = 0;

int *myLabels = labels + (WORK_PER_BLOCK * blockIdx.x);
int *myValues = values + (WORK_PER_BLOCK * blockIdx.x);

for (int i = 0; i < WORK_PER_BLOCK; i += THREADS_PER_BLOCK)
{
	curLabels[threadIdx.x] = myLabels[i + threadIdx.x];
	curValues[threadIdx.x] = myValues[i + threadIdx.x];

__syncthreads();
	if (threadIdx.x == 0)
	{
	#pragma unroll
	for (int j = 0; j < THREADS_PER_BLOCK; j++)
	{
		int curVal = curValues[j];
		int curInd = curLabels[j];
		curValues[j] = buckets[curInd];

		buckets[curInd] += curVal;
	}
	}
__syncthreads();	
	myValues[i + threadIdx.x] = curValues[threadIdx.x];
}

int pitchByInt = pitch / sizeof(int);

for (int i = threadIdx.x; i < BINS; i += THREADS_PER_BLOCK)
{
	allBuckets[(i * pitchByInt) + blockIdx.x] = buckets[i];
}
}

__global__
void naiveGlobalSeparateMultiScan(int *labels, int *values, int *allBuckets, size_t pitch)
{
__shared__ int curLabels[THREADS_PER_BLOCK];
__shared__ int curValues[THREADS_PER_BLOCK];
int *blockBuckets = &allBuckets[blockIdx.x];

int pitchByInt = pitch / sizeof(int);

for (int i = threadIdx.x; i < BINS; i += THREADS_PER_BLOCK)
        blockBuckets[i * pitchByInt] = 0;

int *myLabels = labels + (WORK_PER_BLOCK * blockIdx.x);
int *myValues = values + (WORK_PER_BLOCK * blockIdx.x);

for (int i = 0; i < WORK_PER_BLOCK; i += THREADS_PER_BLOCK)
{
        curLabels[threadIdx.x] = myLabels[i + threadIdx.x];
        curValues[threadIdx.x] = myValues[i + threadIdx.x];

__syncthreads();
        if (threadIdx.x == 0)
        {
        #pragma unroll
        for (int j = 0; j < THREADS_PER_BLOCK; j++)
        {
                int curVal = curValues[j];
                int curInd = curLabels[j];
                curValues[j] = blockBuckets[curInd * pitchByInt];

                blockBuckets[curInd * pitchByInt] += curVal;
        }
        }
__syncthreads();
        myValues[i + threadIdx.x] = curValues[threadIdx.x];
}
}

__global__
void allMultiScan(int *labels, int *values, int *allBuckets, size_t pitch)
{
__shared__ spinerec temp[1024 + BINS];
spinerec *bucket = &temp[1024];

// initialize
int *labelsOffset = &labels[blockIdx.x * 1024];
int *valuesOffset = &values[blockIdx.x * 1024];
#ifdef FETCHLABELS
int myLabels[32];
for (int i = 0; i < 32; i++)
	myLabels[i] = labelsOffset[i * 32 + threadIdx.x];

// MAYBE this should be done by an extra kernel so that we can do it all in parallel
int j = 0;
#endif

#pragma unroll
for (int i = threadIdx.x; i < 1024; i+=32)
{
	temp[i].rowsum=0;
	temp[i].spinesum = 0;
#ifdef FETCHLABELS
	temp[i].spine = 1024 + myLabels[j++];
#else
	temp[i].spine = 1024 + labelsOffset[i];
#endif
}

#pragma unroll
for (int i = threadIdx.x; i < BINS; i+=32)
{
	bucket[i].spine = 1024 + i;
	bucket[i].rowsum = 0;
	bucket[i].spinesum = 0;
}
__syncthreads();

// spinetree phase
#ifdef FETCHLABELS
j = 31;
#endif

#pragma unroll
for (int i = 1024 - 32 + threadIdx.x; i >= 0; i-=32)
{ 
#ifdef FETCHLABELS
	temp[i].spine = bucket[myLabels[j]].spine; 
	bucket[myLabels[j]].spine = i;
	j--;
#else
	temp[i].spine = bucket[labelsOffset[i]].spine;
	bucket[labelsOffset[i]].spine = i;
#endif
}

//rowsums phase

#pragma unroll
for (int i = threadIdx.x * 32; i < threadIdx.x * 32 + 32; i++)
{
	temp[temp[i].spine].rowsum += valuesOffset[i];
}
__syncthreads();

//spinesums phase

#pragma unroll
for (int i = threadIdx.x; i < 1024; i += 32)
{
	temp[temp[i].spine].spinesum = temp[i].spinesum + temp[i].rowsum;
} 
__syncthreads();

//multisums phase

#pragma unroll
for (int i = threadIdx.x * 32; i < threadIdx.x * 32 + 32; i++)
{
	int value = valuesOffset[i];
	valuesOffset[i] = temp[temp[i].spine].spinesum;
if (blockIdx.x == 0 && i < 50)
	printf("index %i gets value %i\n", i, temp[temp[i].spine].spinesum);
	__syncthreads();
	temp[temp[i].spine].spinesum += value;
	__syncthreads();
}

//copy buckets to dram
//int *myBucket = &allBuckets[blockIdx.x]; 
int pitchByInt = pitch / sizeof(int);
#pragma unroll
for (int i = threadIdx.x; i < BINS; i += 32)
{
	allBuckets[(i * pitchByInt) + blockIdx.x] = bucket[i].spinesum;
}

}

void invokeMultiScan(int *indices, int *values, int num_elements)
{
int threadspb = 32;
#ifdef SHEFFLER
int blocks = num_elements / 1024;
#else
int blocks = num_elements / WORK_PER_BLOCK;
#endif 


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

#ifdef SHEFFLER
allMultiScan<<<dimGrid, dimBlock>>>(indices, values, allBuckets, pitch);
#else

//naiveMultiScan<<<blocks, THREADS_PER_BLOCK>>>(indices, values, allBuckets, pitch);
naiveGlobalSeparateMultiScan<<<blocks, THREADS_PER_BLOCK>>>(indices, values, allBuckets, pitch);

#endif

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
int size = 1 << 26;
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


