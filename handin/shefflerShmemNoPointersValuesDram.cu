#include <stdio.h>
#include <sys/time.h>
#include "cudpp.h"

#define BINS 64
#define THREADS_PER_BLOCK 32

#define WORK_PER_BLOCK (THREADS_PER_BLOCK * THREADS_PER_BLOCK)


//#define FETCHLABELS


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

int work = WORK_PER_BLOCK;

int blocks = gridDim.x;

#pragma unroll
for (int i = threadIdx.x; i < BINS; i+=THREADS_PER_BLOCK)
{
	bins[i] = bucketSums[blockIdx.x + i * blocks];
}
__syncthreads();


int *myIndices = &indices[work * blockIdx.x];
int *myResult = &result[work * blockIdx.x];
#pragma unroll
for (int i = threadIdx.x; i < work; i+=THREADS_PER_BLOCK)
{
	myResult[i] += bins[myIndices[i]];
}

}


__global__
void allMultiScan(int *labels, int *values, int *result, spinerec *spinerecs, int *allBuckets, size_t pitch)
{
spinerec *temp = spinerecs + (blockIdx.x * (BINS + WORK_PER_BLOCK));
spinerec *bucket = &temp[WORK_PER_BLOCK];

// initialize
int *labelsOffset = &labels[blockIdx.x * WORK_PER_BLOCK];
int *valuesOffset = &values[blockIdx.x * WORK_PER_BLOCK];
int *resultOffset = &result[blockIdx.x * WORK_PER_BLOCK];

#pragma unroll
for (int i = threadIdx.x; i < WORK_PER_BLOCK; i+=THREADS_PER_BLOCK)
{
	spinerec tempi;
	tempi.rowsum=0;
	tempi.spinesum = 0;
	tempi.spine = WORK_PER_BLOCK + labelsOffset[i];
	temp[i] = tempi;
}
__syncthreads();

#pragma unroll
for (int i = threadIdx.x; i < BINS; i+=THREADS_PER_BLOCK)
{
	spinerec bucketi;
	bucketi.spine = WORK_PER_BLOCK + i;
	bucketi.rowsum = 0;
	bucketi.spinesum = 0;
	bucket[i] = bucketi;
}
__syncthreads();

// spinetree phase

#pragma unroll
for (int i = WORK_PER_BLOCK - THREADS_PER_BLOCK + threadIdx.x; i >= 0; i-=THREADS_PER_BLOCK)
{ 
	temp[i].spine = bucket[labelsOffset[i]].spine;
	__syncthreads();
	bucket[labelsOffset[i]].spine = i;
	__syncthreads();
}

//rowsums phase

#pragma unroll
for (int i = threadIdx.x * THREADS_PER_BLOCK; i < threadIdx.x * THREADS_PER_BLOCK + THREADS_PER_BLOCK; i++)
{
	temp[temp[i].spine].rowsum += valuesOffset[i];
//if (blockIdx.x == 0) printf("thread %i just added %i to spine's rowsum, is now %i\n", threadIdx.x, valuesOffset[i], temp[temp[i].spine].rowsum);
	__syncthreads();
}

//spinesums phase

#pragma unroll
for (int i = threadIdx.x; i < WORK_PER_BLOCK; i += THREADS_PER_BLOCK)
{
if (temp[i].rowsum != 0)
{
	temp[temp[i].spine].spinesum = temp[i].spinesum + temp[i].rowsum;
//	if (blockIdx.x == 0) 
//		printf("thread %i just added rowsum %i to spine's spinesum, is now %i\n", threadIdx.x, temp[i].rowsum, temp[temp[i].spine].spinesum);
}
	__syncthreads();
} 

//multisums phase

#pragma unroll
for (int i = threadIdx.x * THREADS_PER_BLOCK; i < threadIdx.x * THREADS_PER_BLOCK + THREADS_PER_BLOCK; i++)
{
	resultOffset[i] = temp[temp[i].spine].spinesum;
//if (temp[temp[i].spine].spinesum > 700)
//	printf("index %i gets result %i, has value %i and label %i, spine points to %i, own spinesum is %i, rowsum is %i\n", i, temp[temp[i].spine].spinesum, valuesOffset[i], labelsOffset[i], temp[i].spine, temp[i].spinesum, temp[i].rowsum);
	__syncthreads();
//int before = temp[temp[i].spine].spinesum;
	temp[temp[i].spine].spinesum += valuesOffset[i];
//if(blockIdx.x == 0) printf("just added %i to %i, result is %i\n", value, before, temp[temp[i].spine].spinesum);
	__syncthreads();
}

//copy buckets to dram
//int *myBucket = &allBuckets[blockIdx.x]; 
int pitchByInt = pitch / sizeof(int);
#pragma unroll
for (int i = threadIdx.x; i < BINS; i += THREADS_PER_BLOCK)
{
//if (blockIdx.x == 0) printf("bucket %i gets %i\n", i, bucket[i].spinesum);
	allBuckets[(i * pitchByInt) + blockIdx.x] = bucket[i].spinesum;
}

}

void invokeMultiScan(int *indices, int *values, int *result, int num_elements)
{
int threadspb = THREADS_PER_BLOCK;
int blocks = num_elements / WORK_PER_BLOCK;
printf("blocks %i threadspb %i\n", blocks, threadspb);

spinerec *spinerecs;
int *allBuckets;
int *allBucketsResult;
size_t pitch;
size_t pitch_result;
cudaMalloc((void**)&spinerecs, blocks * (BINS + WORK_PER_BLOCK) * sizeof(spinerec));

cudaMallocPitch((void**)&allBuckets, &pitch, blocks * sizeof(int),  BINS);
cudaMallocPitch((void**)&allBucketsResult, &pitch_result, blocks * sizeof(int), BINS);

dim3 dimBlock(threadspb, 1);
dim3 dimGrid(blocks, 1);
cudaFuncSetCacheConfig(allMultiScan, cudaFuncCachePreferShared);
cudaFuncSetCacheConfig(addRemainder, cudaFuncCachePreferShared);


timeval before, between1, between2, after;

gettimeofday(&before, NULL);

allMultiScan<<<dimGrid, dimBlock>>>(indices, values, result, spinerecs, allBuckets, pitch);

cudaThreadSynchronize();


gettimeofday(&between1, NULL);

cudppMS(blocks, allBuckets, allBucketsResult, pitch);
cudaThreadSynchronize();

gettimeofday(&between2, NULL);

addRemainder<<<dimGrid, dimBlock>>>(allBucketsResult, indices, result);


cudaThreadSynchronize();
gettimeofday(&after, NULL);

float time3 = (after.tv_sec - between2.tv_sec) * 1e9 + (after.tv_usec - between2.tv_usec) * 1e3;
float time2 = (between2.tv_sec - between1.tv_sec) * 1e9 + (between2.tv_usec - between1.tv_usec) * 1e3;
float time1 = (between1.tv_sec - before.tv_sec) * 1e9 + (between1.tv_usec - before.tv_usec) * 1e3;
printf("%i\t%i\t%i tbb %f\t%f\t%f\t%f\n", num_elements, BINS, THREADS_PER_BLOCK, time1 / num_elements, time2 / num_elements, time3 / num_elements, (time1 + time2 + time3) / num_elements);
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
//printf("spinerec hsa size %i\n", sizeof(spinerec));

int size = ((1 << 26) / WORK_PER_BLOCK ) * WORK_PER_BLOCK;
//int size = WORK_PER_BLOCK;
int *keys = (int*) malloc(size * sizeof(int));
int *values = (int*) malloc(size * sizeof(int));
int *result = (int*) malloc(size * sizeof(int));
int *result_cpu = (int*) malloc(size * sizeof(int));
int index = rand() % BINS; 

for (int i = 0; i < size; i++)
{
	values[i] = rand() % 10 + 1;
	keys[i] = rand() % BINS;
//if (i < 5)
//	printf("original value %i %i \n", i, values[i]);
//	result[i] = 0;
}

int *d_keys, *d_values, *d_result;
cudaMalloc((void**) &d_result, size * sizeof(int));
cudaMalloc((void**) &d_keys, size * sizeof(int));
cudaMalloc((void**) &d_values, size * sizeof(int));

cudaMemcpy(d_keys, keys, size * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, size * sizeof(int), cudaMemcpyHostToDevice);

invokeMultiScan(d_keys, d_values, d_result, size);

cudaMemcpy(result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

timeval before, after;
gettimeofday(&before, NULL);
multiScanCpu(keys, values, result_cpu, size);
gettimeofday(&after, NULL);

printf("cpu needed %f ns/int \n", ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / size);

for (int i = 0; i < size; i++)
{
//	if (result[i] != result_cpu[i])
//		printf("I sense a discrepancy! index %i gpu %i corect %i value %i\n", i, result[i], result_cpu[i], values[i]);
//	else
//		printf("correct! %i %i %i\n", i, result[i], result_cpu[i]);
}

printf("last error: %i\n", cudaGetLastError());
}


