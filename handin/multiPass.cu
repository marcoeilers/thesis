#define THREADS_PER_BLOCK 512

__global__ void getOffsetsKernel(int *indices, int *result, int todopb, int todoTotal)
{
__shared__ int currentNums[THREADS_PER_BLOCK + 1];

int start = blockIdx.x * todopb;
int *ourIndices = &indices[start];

todopb = min(todoTotal - start, todopb);

int lastIndex = start - 1;
currentNums[0] = lastIndex < 0 ? -1 : indices[lastIndex];

for (int i = threadIdx.x; i < todopb; i += THREADS_PER_BLOCK)
{
	currentNums[threadIdx.x + 1] = ourIndices[i];
	__syncthreads();

	int first = currentNums[threadIdx.x];
	int second = currentNums[threadIdx.x + 1];
	
	if (first >> 8 != second >> 8)
		result[second >> 8] = start + i;
	__syncthreads();
}
 
}


__host__ void getOffsets(int *indices, int *result, int num_elements)
{
int blocks = 8 * 16;

int todopb = num_elements / blocks;

todopb = todopb * blocks >= num_elements ? todopb : todopb + 1;

cudaFuncSetCacheConfig(getOffsetsKernel, cudaFuncCachePreferShared);

getOffsetsKernel<<<blocks, THREADS_PER_BLOCK>>>(indices, result, todopb, num_elements);
}
