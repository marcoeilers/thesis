#include <stdio.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>

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
__shared__ int ourHistsInt[BINS * WARP_SIZE / sizeof(int)];
//int *ourHistsInt = (int*) &ourHists[0];
int curInd = threadIdx.x;
int *ourIndices = &indices[(todopb / 4) * blockIdx.x];
int *myHistsInt = &ourHistsInt[threadIdx.x % WARP_SIZE];
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
			int curByte = (curVal >> (k * 8)) & 255;
			atomicAdd(&myHistsInt[(curByte >> 2 << 5)], 1 << ((curByte % 4) << 3));
		}

		curInd += THREADS_PER_BLOCK;
	}
	
	__syncthreads();
	
	// now accumulate stuff in registers. 
	// warp 0 gets bins 0 and 2 of ints 0-31, so every second of 0-127
	// warp 1 gets bins 0 and 2 of ints 32-63, so every second of 128-255
	// warp 2 gets bins 1 and 3 of ints 0-31

	int myValue = 0;
	int j = threadIdx.x % WARP_SIZE;
	int offset = WARP_SIZE * (threadIdx.x % (WARP_SIZE * 2));
	if (threadIdx.x >= (2 * WARP_SIZE))
	{
		#pragma unroll 32
		do {
			int curVal = ourHistsInt[offset + j];
			myValue += curVal & 16711935;
			j = (j + 1) % WARP_SIZE;
		} while (j != threadIdx.x % WARP_SIZE);
	}else{
		#pragma unroll 32
		do {
			int curVal = ourHistsInt[offset + j];
			myValue += (curVal & 4278255360) >> 8;
			j = (j + 1) % WARP_SIZE;
		} while (j != threadIdx.x % WARP_SIZE);
	}
	myVal1 += myValue & 65535;
	myVal2 += (myValue >> 16) & 65535;
	__syncthreads();

	int start = threadIdx.x % WARP_SIZE + (threadIdx.x / WARP_SIZE) * 16 * WARP_SIZE;
	for (int j = start; j < start + 16 * WARP_SIZE; j += WARP_SIZE)
		ourHistsInt[j] = 0;
	__syncthreads();
}

// write my two values to global memory
if (threadIdx.x >= (2 * WARP_SIZE))
{
//result[(threadIdx.x % (WARP_SIZE * 2)) * 4] += myVal1;
//result[(threadIdx.x % (WARP_SIZE * 2)) * 4 + 2] += myVal2;
atomicAdd(&result[(threadIdx.x % (WARP_SIZE * 2)) * 4], myVal1);
atomicAdd(&result[(threadIdx.x % (WARP_SIZE * 2)) * 4 + 2], myVal2);
}else{
//result[(threadIdx.x % (WARP_SIZE * 2)) * 4 + 1] += myVal1;
//result[(threadIdx.x % (WARP_SIZE * 2)) * 4 + 3] += myVal2;
atomicAdd(&result[(threadIdx.x % (WARP_SIZE * 2)) * 4 + 1], myVal1);
atomicAdd(&result[(threadIdx.x % (WARP_SIZE * 2)) * 4 + 3], myVal2);
}

}

void bankTRISH(int *indices, int *result, int num_elements)
{
cudaMemset(result, 0, BINS * sizeof(int));
int todopb = num_elements / 48;

if (todopb % BLOCK_BYTES_PER_RUN != 0)
	printf("ERROR: todopb is %i, block bytes per run is %i\n", todopb, BLOCK_BYTES_PER_RUN);

if (num_elements % 48 != 0)
	printf("ERROR: num_elements is %i\n", num_elements);

cudaProfilerStart();
bankTRISHKernel<<<48, THREADS_PER_BLOCK>>>(indices, result, num_elements, todopb);
cudaThreadSynchronize();
cudaProfilerStop();
printf("cuda last error is %i\n", cudaGetLastError());
}

void histCPU(unsigned char *indices, int *result, int num_elements)
{
	for (int i = 0; i < BINS; i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
	{
		int curInd = indices[i];
		result[curInd]++;
		if (i < 100)
		printf("value, %i\n", curInd);
	}
}

int main()
{
int num_elements = 48 * 7680 * 192;

unsigned char *indices = (unsigned char*) malloc(num_elements);
int *result = (int*) malloc(BINS * sizeof(int));
int *result_cpu = (int*) malloc(BINS * sizeof(int));

for (int i = 0; i < num_elements; i++)
{
	indices[i] = rand() % 256;
//	indices[i] = 1;
}

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
histCPU(indices, result_cpu, num_elements);

for (int i = 0; i < BINS; i++)
{
	if (result[i] != result_cpu[i])
		printf("I sense a discrepancy! %i %i %i\n", i, result[i], result_cpu[i]);
//	else
//		printf("index %i seems correct\n", i);
}

printf("ns / int %f\n", ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);
}
