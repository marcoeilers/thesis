#include <stdio.h>
#include <sys/time.h>

#define EXP 26
#define THREADS_PER_BLOCK 32
#define BUCKETS 64

__global__ void multiReduceKernel(int *indices, int *values, int *result, int todopb)
{
__shared__ int blockBuckets[THREADS_PER_BLOCK * BUCKETS];

// initialize buckets to zero
for (int i = threadIdx.x; i < THREADS_PER_BLOCK * BUCKETS; i += THREADS_PER_BLOCK)
	blockBuckets[i] = 0;

int *blockIndices = indices + todopb * blockIdx.x;
int *blockValues = values + todopb * blockIdx.x;

int *myBuckets = &blockBuckets[threadIdx.x];

for (int i = threadIdx.x; i < todopb; i += THREADS_PER_BLOCK)
{
	int curInd = blockIndices[i];
	int curVal = blockValues[i];

	myBuckets[curInd * THREADS_PER_BLOCK] += curVal;
}

//copy results back to DRAM
int i = threadIdx.x;
do {
	atomicAdd(&result[i], myBuckets[i * THREADS_PER_BLOCK]);

	i = (i + 1) % BUCKETS; 
} while(i != threadIdx.x);

}

__host__ void multiReduce(int *indices, int *values, int *result, int num_elements)
{

// set bank width to 32 bit
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
// prefer shared mem
cudaFuncSetCacheConfig(multiReduceKernel, cudaFuncCachePreferShared);

cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);


int shmemPerBlock = THREADS_PER_BLOCK * BUCKETS * sizeof(int);
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

printf("ns/int %f\n", ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);

for (int i = 0; i < BUCKETS; i++)
{
	if (result[i] != result_cpu[i])
		printf("i sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
//	else
//		printf("it's all good!\n");
}

}
