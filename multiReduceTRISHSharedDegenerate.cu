#include <stdio.h>
#include <sys/time.h>

#define EXP 26
#define HISTS_PER_BLOCK 32
#define THREADS_PER_BLOCK 64
#define BUCKETS 8

__global__ void multiReduceKernel(int *indices, int *values, int *result, int todopb)
{
__shared__ int blockBuckets[HISTS_PER_BLOCK * BUCKETS];

// initialize buckets to zero
for (int i = threadIdx.x; i < HISTS_PER_BLOCK * BUCKETS; i += THREADS_PER_BLOCK)
	blockBuckets[i] = 0;

__syncthreads();

int *blockIndices = indices + todopb * blockIdx.x;
int *blockValues = values + todopb * blockIdx.x;

int *myBuckets = &blockBuckets[threadIdx.x % HISTS_PER_BLOCK];

int i;
for (i = threadIdx.x; i + 3 * THREADS_PER_BLOCK < todopb; i += 4 * THREADS_PER_BLOCK)
{
	int curInd = blockIndices[i];
	int curVal = blockValues[i];

        int curInd2 = blockIndices[i + THREADS_PER_BLOCK];
        int curVal2 = blockValues[i + THREADS_PER_BLOCK];

	curVal = curVal + (curInd == curInd2) * curVal2;
	curVal2 = (curInd != curInd2) * curVal2;
	curInd2 = (curInd2 + (curInd == curInd2) * threadIdx.x) % BUCKETS;

	
	atomicAdd(&myBuckets[curInd * HISTS_PER_BLOCK], curVal);
	atomicAdd(&myBuckets[curInd2 * HISTS_PER_BLOCK], curVal2);

        int curInd3 = blockIndices[i + 2 * THREADS_PER_BLOCK];
        int curVal3 = blockValues[i + 2 * THREADS_PER_BLOCK];

        int curInd4 = blockIndices[i + 3 * THREADS_PER_BLOCK];
        int curVal4 = blockValues[i + 3 * THREADS_PER_BLOCK];

        curVal3 = curVal3 + (curInd3 == curInd4) * curVal4;
        curVal4 = (curInd3 != curInd4) * curVal4;
        curInd4 = (curInd4 + (curInd3 == curInd4) * threadIdx.x) % BUCKETS;


        atomicAdd(&myBuckets[curInd3 * HISTS_PER_BLOCK], curVal3);
        atomicAdd(&myBuckets[curInd4 * HISTS_PER_BLOCK], curVal4);
}

while (i < todopb)
{
        int curInd = blockIndices[i];
        int curVal = blockValues[i];
        atomicAdd(&myBuckets[curInd * HISTS_PER_BLOCK], curVal);
	i += THREADS_PER_BLOCK;
}
__syncthreads();


//copy results back to DRAM
if (threadIdx.x < HISTS_PER_BLOCK)
{
int i = threadIdx.x % BUCKETS;
do {
	atomicAdd(&result[i], myBuckets[i * HISTS_PER_BLOCK]);

	i = (i + 1) % BUCKETS; 
} while(i != threadIdx.x % BUCKETS);
}

}

__host__ void multiReduce(int *indices, int *values, int *result, int num_elements)
{

// set bank width to 32 bit
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
// prefer shared mem
cudaFuncSetCacheConfig(multiReduceKernel, cudaFuncCachePreferShared);

cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);


int shmemPerBlock = HISTS_PER_BLOCK * BUCKETS * sizeof(int);
int blocksPerSm = props.sharedMemPerBlock / shmemPerBlock;

int totalBlocks = props.multiProcessorCount * blocksPerSm;
printf("%i SMs, %i bytes shared memory, we use %i bytes per block, therefore have %i blocks\n", props.multiProcessorCount, props.sharedMemPerBlock, shmemPerBlock, totalBlocks);

multiReduceKernel<<<totalBlocks, THREADS_PER_BLOCK>>>(indices, values, result, num_elements / totalBlocks);

}

void multiReduceCpu(int *keys, int *values, int *result, int num_elements)
{
	for (int i = 0; i < BUCKETS; i++)
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
result = (int*) calloc(BUCKETS, sizeof(int));
result_cpu = (int*) calloc(BUCKETS, sizeof(int));

for (int i = 0; i < num_elements; i++)
{
	indices[i] = rand() % BUCKETS;
	values[i] = rand() % 1000;
}

int *d_keys, *d_values, *d_result;
cudaMalloc((void**)&d_keys, num_elements * sizeof(int));
cudaMalloc((void**)&d_values, num_elements * sizeof(int));
cudaMalloc((void**)&d_result, BUCKETS * sizeof(int));

cudaMemcpy(d_keys, indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, num_elements * sizeof(int), cudaMemcpyHostToDevice);
cudaMemset(d_result, 0, BUCKETS * sizeof(int));


timeval before, after;
gettimeofday(&before, NULL);
multiReduce(d_keys, d_values, d_result, num_elements);
cudaThreadSynchronize();
gettimeofday(&after, NULL);

cudaMemcpy(result, d_result, BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);
multiReduceCpu(indices, values, result_cpu, num_elements);

printf("%i buckets, ns/int %f\n", BUCKETS, ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);

for (int i = 0; i < BUCKETS; i++)
{
	if (result[i] != result_cpu[i])
		printf("i sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
//	else
//		printf("it's all good!\n");
}

}
