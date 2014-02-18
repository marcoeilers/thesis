#include <stdio.h>
#include <sys/time.h>

#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)
#define BINS 256
#define INTS_PER_RUN ((BINS - 1) / (sizeof(int) * WARPS_PER_BLOCK))
#define BLOCK_BYTES_PER_RUN (INTS_PER_RUN * 4 * THREADS_PER_BLOCK)

/*
 * num_elements and todopb in bytes
 */
__global__ void bankTRISHKernel(int *indices, int *result, int num_elements, int todopb)
{
__shared__ char ourHists[BINS * WARP_SIZE];
int *ourHistsInt = (int*) &ourHists[0];
int curInd = threadIdx.x;
int *ourIndices = &indices[todopb * blockIdx.x];
int myVal1 = 0;
int myVal2 = 0;

for (int i = 0; i < todopb; i += BLOCK_BYTES_PER_RUN)
{
	#pragma unroll
	for (int j = 0; j < INTS_PER_RUN; j++)
	{
		int curVal = ourIndices[curInd];
		
		#pragma unroll
		for (int k = 0; k < 4; k++)
		{
			char curByte = (curVal >> (k * 8)) & 255;
//			ourHistsInt[(curByte >> 2) * WARP_SIZE + (threadIdx.x % WARP_SIZE)] += curByte;
			atomicAdd(&ourHistsInt[(curByte >> 2) * WARP_SIZE + (threadIdx.x % WARP_SIZE)], curByte);
		}

		curInd += THREADS_PER_BLOCK;
	}
	
	// now accumulate stuff in registers. 
	// warp 0 gets bins 0 and 2 of ints 0-31
	int myValue = 0;
	int j = threadIdx.x % (WARP_SIZE << 1);
	if ((threadIdx.x / WARP_SIZE) / (WARP_SIZE << 1) == 0)
	{
		do {
			int curVal = ourHistsInt[j + (WARP_SIZE * (threadIdx.x % (WARP_SIZE << 1)))];
			myVal += curVal & 16711935;
			j = j + 1 % (WARP_SIZE << 2);
		} while (j != threadIdx.x);
	}else{
		do {
			int curVal = ourHistsInt[j + (WARP_SIZE * (threadIdx.x % (WARP_SIZE << 1)))];
			myVal += (curVal & 4278255360) >> 8;
			j = j + 1 % (WARP_SIZE << 2);
		} while (j != threadIdx.x);
	}
	myVal1 += myValue & 65535;
	myVal2 += (myValue >> 16) & 65535;
}

// write my two values to global memory
atomicAdd(&result[threadIdx.x * 4], myVal1);
atomicAdd(&result[threadIdx.x * 4 + 2], myVal2);

}

void bankTRISH(int *indices, int *result, int num_elements)
{
cudaMemset(result, 0, BINS * sizeof(int));
int blocks = 48;
int todopb = num_elements / 48;
bankTRISHKernel<<<48, THREADS_PER_BLOCK>>>(indices, result, num_elements, todopb);
cudaThreadSynchronize();
}

void histCPU(char *indices, int *result, int num_elements)
{
	for (int i = 0; i < BINS; i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
		result[indices[i]]++;
}

int main()
{
int num_elements = 48 * 7680 * 192;

char *indices = (char*) malloc(num_elements);
int *result = (int*) malloc(BINS * sizeof(int));
int *result_cpu = (int*) malloc(BINS * sizeof(int));

char *d_indices; 
int *d_result;
cudaMalloc((void**)&d_indices, num_elements);
cudaMalloc((void**)&d_result, BINS * sizeof(int));

cudaMemcpy(d_indices, indices, num_elements, cudaMemcpyHostToDevice);

timeval before, after;

gettimeofday(&before, NULL);
bankTRISH((int*)d_indices, d_result, num_elements);
gettimeofday(&after, NULL);

cudaMemcpy(result, d_result, BINS * sizeof(int), cudaMemcpyDeviceToHost);
histCPU(indices, result_cpu, num_elements)

for (int i = 0; i < BINS; i++)
{
	if (result[i] != result_cpu[i])
		printf("I sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
}

printf("ns / int %f\n", ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);
}
