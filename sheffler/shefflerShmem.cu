#include <stdio.h>
#include <sys/time.h>
#include "cudpp.h"

#define BINS 64

//#define FETCHLABELS

//TODO bucket results are stored in the wrong order
// which doesnt influence performance i think but makes results wrong

struct spinerec {
	int rowsum;
	int spinesum;
	int spine;
};
void cudppMS(int blocks, int *in, int *out);


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


int *myIndices = &indices[1024 * blockIdx.x];
int *myResult = &result[1024 * blockIdx.x];
#pragma unroll
for (int i = threadIdx.x; i < 1024; i+=32)
{
	myResult[i] += bins[myIndices[i]];
}

}


__global__
void allMultiScan(int *labels, int *values, int *allBuckets)
{
__shared__ spinerec temp[1024 + BINS];
//__shared__ spinerec bucket[BINS];
spinerec *bucket = &temp[1024];

// initialize
int *labelsOffset = &labels[blockIdx.x * 1024];
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
//__syncthreads();

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
	temp[temp[i].spine].rowsum += values[i];
}
//__syncthreads();

//spinesums phase

#pragma unroll
for (int i = threadIdx.x; i < 1024; i += 32)
{
	temp[temp[i].spine].spinesum = temp[i].spinesum + temp[i].rowsum;
} 
//__syncthreads();

//multisums phase

#pragma unroll
for (int i = threadIdx.x * 32; i < threadIdx.x * 32 + 32; i++)
{
	values[i] = temp[temp[i].spine].spinesum;
//	__syncthreads();
	temp[temp[i].spine].spinesum += values[i];
//	__syncthreads();
}

//copy buckets to dram
int *myBucket = &allBuckets[blockIdx.x * BINS]; 

#pragma unroll
for (int i = threadIdx.x; i < BINS; i += 32)
{
	myBucket[i] = bucket[i].spinesum;
}
	
}

void invokeMultiScan(int *indices, int *values, int num_elements)
{
int threadspb = 32;
int blocks = num_elements / 1024;

int *allBuckets;
cudaMalloc((void**)&allBuckets, 2 * BINS * blocks * sizeof(int));

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);
cudaFuncSetCacheConfig(allMultiScan, cudaFuncCachePreferShared);

timeval before, after;
gettimeofday(&before, NULL);
allMultiScan<<<dimGrid, dimBlock>>>(indices, values, allBuckets);
cudaThreadSynchronize();

//gettimeofday(&before, NULL);
cudppMS(blocks, allBuckets, allBuckets + (BINS*blocks));

addRemainder<<<dimGrid, dimBlock>>>(allBuckets + (BINS*blocks), indices, values);

cudaThreadSynchronize();
gettimeofday(&after, NULL);

printf("%i\t%f\n", num_elements, ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);
}

void cudppMS(int blocks, int *in, int *out)
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
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, blocks, BINS, 0);  
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

int main()
{
printf("spinerec hsa size %i\n", sizeof(spinerec));
int size = 1 << 26;
int *keys = (int*) malloc(size * sizeof(int));
int *values = (int*) malloc(size * sizeof(int));
int *result = (int*) malloc(size * sizeof(int));

for (int i = 0; i < size; i++)
{
	values[i] = rand() % 1000;
	keys[i] = rand() % BINS;
	result[i] = 0;
}

int *d_keys, *d_values;
cudaMalloc((void**) &d_keys, size * sizeof(int));
cudaMalloc((void**) &d_values, size * sizeof(int));

cudaMemcpy(d_keys, keys, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, size * sizeof(int), cudaMemcpyHostToDevice);

invokeMultiScan(d_keys, d_values, size);

printf("last error: %i\n", cudaGetLastError());
}


