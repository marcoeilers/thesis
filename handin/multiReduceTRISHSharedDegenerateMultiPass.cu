#include <stdio.h>
#include <sys/time.h>

#define EXP 26
#define HISTS_PER_BLOCK 32
#define BUCKETS 256
#define TB_BITS 8
#define TOTALBUCKETS (1 << TB_BITS)
#define THREADS_PER_BLOCK (BUCKETS * 4)
#define NO_RUNS 8

//#define DEGENERATE

__global__ void multiReducePassKernel(int *indices, int *values, int *result, int firstBucket, int todopb)
{
__shared__ int blockBuckets[HISTS_PER_BLOCK * BUCKETS];
int firstNextBucket = firstBucket + BUCKETS;

// initialize buckets to zero
for (int i = threadIdx.x; i < HISTS_PER_BLOCK * BUCKETS; i += THREADS_PER_BLOCK)
	blockBuckets[i] = 0;

__syncthreads();

int *blockIndices = indices + todopb * blockIdx.x;
int *blockValues = values + todopb * blockIdx.x;

int *myBuckets = &blockBuckets[threadIdx.x % HISTS_PER_BLOCK];

int i;

#pragma unroll 16
for (i = threadIdx.x; i + 3 * THREADS_PER_BLOCK < todopb; i += 4 * THREADS_PER_BLOCK)
{
	int curInd = blockIndices[i];
	int curVal = blockValues[i];

        int curInd2 = blockIndices[i + THREADS_PER_BLOCK];
        int curVal2 = blockValues[i + THREADS_PER_BLOCK];

        int curInd3 = blockIndices[i + 2 * THREADS_PER_BLOCK];
        int curVal3 = blockValues[i + 2 * THREADS_PER_BLOCK];

        int curInd4 = blockIndices[i + 3 * THREADS_PER_BLOCK];
        int curVal4 = blockValues[i + 3 * THREADS_PER_BLOCK];

	curVal = curVal + (curInd == curInd2) * curVal2;
//	curVal2 = (curInd != curInd2) * curVal2;
//	curInd2 = (curInd2 + (curInd == curInd2) * threadIdx.x) % BUCKETS;
	
	if (curInd >= firstBucket && curInd < firstNextBucket)
	atomicAdd(&myBuckets[(curInd % BUCKETS) * HISTS_PER_BLOCK], curVal);
	if (curInd2 >= firstBucket && curInd2 < firstNextBucket && curInd != curInd2)
	atomicAdd(&myBuckets[(curInd2 % BUCKETS) * HISTS_PER_BLOCK], curVal2);

        curVal3 = curVal3 + (curInd3 == curInd4) * curVal4;
//        curVal4 = (curInd3 != curInd4) * curVal4;
//        curInd4 = (curInd4 + (curInd3 == curInd4) * threadIdx.x) % BUCKETS;

	if (curInd3 >= firstBucket && curInd3 < firstNextBucket)
        atomicAdd(&myBuckets[(curInd3 % BUCKETS) * HISTS_PER_BLOCK], curVal3);
	if (curInd4 >= firstBucket && curInd4 < firstNextBucket && curInd3 != curInd4)
        atomicAdd(&myBuckets[(curInd4 % BUCKETS) * HISTS_PER_BLOCK], curVal4);
}

while (i < todopb)
{
        int curInd = blockIndices[i];
        int curVal = blockValues[i];
	if (curInd >= firstBucket && curInd < firstNextBucket)
        atomicAdd(&myBuckets[(curInd % BUCKETS) * HISTS_PER_BLOCK], curVal);
	i += THREADS_PER_BLOCK;
}
__syncthreads();


//copy results back to DRAM
if (threadIdx.x < HISTS_PER_BLOCK)
{
int i = threadIdx.x % BUCKETS;
do {
	atomicAdd(&result[i], myBuckets[i * HISTS_PER_BLOCK]);
//	result[i] += myBuckets[i * HISTS_PER_BLOCK];
	i = (i + 1) % BUCKETS; 
} while(i != threadIdx.x % BUCKETS);
}

}

__host__ void multiReduce(int *indices, int *values, int *result, int num_elements)
{
cudaMemset(result, 0, TOTALBUCKETS * sizeof(int));

// set bank width to 32 bit
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
// prefer shared mem
cudaFuncSetCacheConfig(multiReducePassKernel, cudaFuncCachePreferShared);

cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);


int shmemPerBlock = HISTS_PER_BLOCK * BUCKETS * sizeof(int);
int blocksPerSm = props.sharedMemPerBlock / shmemPerBlock;

int totalBlocks = props.multiProcessorCount * blocksPerSm;
//printf("%i SMs, %i bytes shared memory, we use %i bytes per block, therefore have %i blocks\n", props.multiProcessorCount, props.sharedMemPerBlock, shmemPerBlock, totalBlocks);

for (int start = 0; start < TOTALBUCKETS; start += BUCKETS)
multiReducePassKernel<<<totalBlocks, THREADS_PER_BLOCK>>>(indices, values, &result[start], start, num_elements / totalBlocks);

}

void multiReduceCpu(int *keys, int *values, int *result, int num_elements)
{
	for (int i = 0; i < TOTALBUCKETS; i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
	{
		result[keys[i]] += values[i];
	}
}

int main()
{
int num_elements = (1 << (EXP - 1)) * 3; 
int *indices, *values, *result, *result_cpu;

indices = (int*) calloc(num_elements, sizeof(int));
values = (int*) calloc(num_elements, sizeof(int));
result = (int*) calloc(TOTALBUCKETS, sizeof(int));
result_cpu = (int*) calloc(TOTALBUCKETS, sizeof(int));

int *d_keys, *d_values, *d_result;
cudaMalloc((void**)&d_keys, num_elements * sizeof(int));
cudaMalloc((void**)&d_values, num_elements * sizeof(int));
cudaMalloc((void**)&d_result, TOTALBUCKETS * sizeof(int));



{


for (int i = 0; i < num_elements; i++)
{
        indices[i] = rand() % TOTALBUCKETS;
        values[i] = rand() % 1000;
}

cudaMemcpy(d_keys, indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, num_elements * sizeof(int), cudaMemcpyHostToDevice);
//cudaMemset(d_result, 0, TOTALBUCKETS * sizeof(int));


timeval before, after;
gettimeofday(&before, NULL);
for (int i = 0; i < NO_RUNS; i++)
	multiReduce(d_keys, d_values, d_result, num_elements);
cudaThreadSynchronize();
gettimeofday(&after, NULL);

cudaMemcpy(result, d_result, BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);
multiReduceCpu(indices, values, result_cpu, num_elements);

printf("Random %i buckets, ns/int %f\n", TOTALBUCKETS, ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / (num_elements * NO_RUNS));

for (int i = 0; i < BUCKETS; i++)
{
	if (result[i] != result_cpu[i])
		printf("i sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
}
}
{

int index = rand() % TOTALBUCKETS;

for (int i = 0; i < num_elements; i++)
{
        indices[i] = index;
        values[i] = rand() % 1000;
}

cudaMemcpy(d_keys, indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, num_elements * sizeof(int), cudaMemcpyHostToDevice);
//cudaMemset(d_result, 0, TOTALBUCKETS * sizeof(int));


timeval before, after;
gettimeofday(&before, NULL);
for (int i = 0; i < NO_RUNS; i++)
	multiReduce(d_keys, d_values, d_result, num_elements);
cudaThreadSynchronize();
gettimeofday(&after, NULL);

cudaMemcpy(result, d_result, BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);
multiReduceCpu(indices, values, result_cpu, num_elements);

printf("Deg %i buckets, ns/int %f\n", BUCKETS, ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / (num_elements * NO_RUNS));
for (int i = 0; i < BUCKETS; i++)
{
        if (result[i] != result_cpu[i])
                printf("i sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
}

}

}
