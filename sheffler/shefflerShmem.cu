#include <stdio.h>
#include <sys/time.h>
#include "cudpp.h"

#define BINS 64

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

int blocks = gridDim.x;

#pragma unroll
for (int i = threadIdx.x; i < BINS; i+=32)
{
	bins[i] = bucketSums[blockIdx.x + i * blocks];
}
__syncthreads();
//if (blockIdx.x == 0)
//	printf("in addremainder first few bin %i has value %i\n", threadIdx.x, bins[threadIdx.x]);


int *myIndices = &indices[1024 * blockIdx.x];
int *myResult = &result[1024 * blockIdx.x];
#pragma unroll
for (int i = threadIdx.x; i < 1024; i+=32)
{
	myResult[i] += bins[myIndices[i]];
}

}


__global__
void allMultiScan(int *labels, int *values, int *allBuckets, size_t pitch)
{
//if (blockIdx.x == 0 && threadIdx.x < 5)
//	printf("in allmultiscan, value %i is %i\n", threadIdx.x, values[threadIdx.x]);
__shared__ spinerec temp[1024 + BINS];
//__shared__ spinerec bucket[BINS];
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
//	if (threadIdx.x == 0)
//		printf("rowsums phase temp %i value %i new rowsum %i\n", i, values[i], temp[temp[i].spine].rowsum);
}
__syncthreads();

//spinesums phase

#pragma unroll
for (int i = threadIdx.x; i < 1024; i += 32)
{
	temp[temp[i].spine].spinesum = temp[i].spinesum + temp[i].rowsum;
//	if (threadIdx.x == 0)
//		printf("thread 0 spinesums phase temp %i spinesum %i rowsum %i\n", i, temp[i].spinesum, temp[i].rowsum);
} 
__syncthreads();

//multisums phase

#pragma unroll
for (int i = threadIdx.x * 32; i < threadIdx.x * 32 + 32; i++)
{
	int value = valuesOffset[i];
	valuesOffset[i] = temp[temp[i].spine].spinesum;
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
//	if (threadIdx.x == 0)
//		printf("thread 0 bucket %i spinesum %i\n", i, bucket[i].spinesum);
}

}

void invokeMultiScan(int *indices, int *values, int num_elements)
{
int threadspb = 32;
int blocks = num_elements / 1024;

int *allBuckets;
int *allBucketsResult;
size_t pitch;
size_t pitch_result;
cudaMallocPitch((void**)&allBuckets, &pitch, blocks * sizeof(int),  BINS);
cudaMallocPitch((void**)&allBucketsResult, &pitch_result, blocks * sizeof(int), BINS);

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);
cudaFuncSetCacheConfig(allMultiScan, cudaFuncCachePreferShared);

//printf("before multiscan indices values\n");
//printDeviceArrays(indices, values, 100);
//printf("in invokemultiscan values address is %u\n", values);
//printDeviceArrays(values, values, 1);
timeval before, after;
gettimeofday(&before, NULL);
allMultiScan<<<dimGrid, dimBlock>>>(indices, values, allBuckets, pitch);
cudaThreadSynchronize();

//printf("\n\n\nafter multiscan indices result\n");
//printDeviceArrays(indices, values, 100);


//gettimeofday(&before, NULL);
cudppMS(blocks, allBuckets, allBucketsResult, pitch);

//printf("\n\n\n\nbuckets before and after scan\n");
//printDeviceArrays(allBuckets, allBucketsResult, blocks * 2);

addRemainder<<<dimGrid, dimBlock>>>(allBucketsResult, indices, values);

cudaThreadSynchronize();
gettimeofday(&after, NULL);

printf("%i\t%f\n", num_elements, ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);
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
 
for (int i = 0; i < size; i++)
{
	values[i] = rand() % 100;
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

multiScanCpu(keys, values, result_cpu, size);

for (int i = 0; i < size; i++)
{
	if (result[i] != result_cpu[i])
		printf("I sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
	else
		printf("correct!\n");
}

printf("last error: %i\n", cudaGetLastError());
}


