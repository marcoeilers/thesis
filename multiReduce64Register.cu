#include <stdio.h>
#include <sys/time.h>

#define EXP 26
#define BINS_PER_THREAD 16
#define SHARED_HIST_SIZE 64
#define NUMS_PER_GO (SHARED_HIST_SIZE / BINS_PER_THREAD)

__host__ void multiReduceCpu(int *keys, int *values, int *result, int num_elements)
{
	for (int i = 0; i < SHARED_HIST_SIZE; i++)
		result[i] = 0;

	for (int i = 0; i < num_elements; i++)
	{
		result[keys[i]] += values[i];
	}
}


inline __device__ void putInHist(int *partHist, int absIndex, int value)
{
	int index = absIndex % BINS_PER_THREAD;

	#pragma unroll	
	for (int i = 0; i < BINS_PER_THREAD; i++)
	{
		partHist[i] += value * (index == i);
	}
}

__global__ void computePartMRKernel(int *keys, int *values, int *hist, int todo)
{
	__shared__ int sharedHist[SHARED_HIST_SIZE];
	__shared__ int sharedValues[SHARED_HIST_SIZE];

	int myPartHist[BINS_PER_THREAD];

	#pragma unroll
	for (int i = 0; i < BINS_PER_THREAD; i++)
	{
		myPartHist[i] = 0;
	}
	__syncthreads();	
	int myStartIndex = (threadIdx.x % NUMS_PER_GO) * BINS_PER_THREAD;

	int *blockValues = values + blockIdx.x * todo;
	int *blockKeys = keys + blockIdx.x * todo;

	for (int i = threadIdx.x; i < todo; i += blockDim.x)
	{
		sharedHist[threadIdx.x] = blockKeys[i];
		sharedValues[threadIdx.x] = blockValues[i];

		__syncthreads();

		int currentIndex = threadIdx.x / NUMS_PER_GO;
		#pragma unroll
		for (int j = 0; j < NUMS_PER_GO; j++)
		{
			int currentKey = sharedHist[currentIndex];

//			myPartHist[currentKey % BINS_PER_THREAD] += sharedValues[currentIndex] * (currentKey >= myStartIndex && currentKey < (myStartIndex + BINS_PER_THREAD));
			if (currentKey >= myStartIndex && currentKey < (myStartIndex + BINS_PER_THREAD))
			{
			        int index = currentKey % BINS_PER_THREAD;
				int value = sharedValues[currentIndex];
			        #pragma unroll
			        for (int i = 0; i < BINS_PER_THREAD; i++)
			        {
			                myPartHist[i] += value * (index == i);
//					if (index == 0)
//						printf("thread %i, currentKey %i, value %i, currenindex %i\n", threadIdx.x, currentKey, value, currentIndex);
			        }

			}
			currentIndex += BINS_PER_THREAD;
		}

		__syncthreads();
	}
//if (myPartHist[0] > 0)
//	printf("thread %i has value %i on zero\n", threadIdx.x, myPartHist[0]);
	//aggregate partial histograms in shared memory
	//clear shared mem histogram first
	sharedHist[threadIdx.x] = 0;
	__syncthreads();
/*
	#pragma unroll
	for (int i = threadIdx.x; i < threadIdx.x + BINS_PER_THREAD; i++)
	{
		atomicAdd(&sharedHist[(threadIdx.x % NUMS_PER_GO) * BINS_PER_THREAD + (i % BINS_PER_THREAD)], myPartHist[(i % BINS_PER_THREAD)]);
	}
*/
        #pragma unroll
        for (int i = 0; i < BINS_PER_THREAD; i++)
                atomicAdd(&sharedHist[(threadIdx.x % NUMS_PER_GO * BINS_PER_THREAD) + i], myPartHist[i]);

	__syncthreads();
	atomicAdd(&hist[threadIdx.x], sharedHist[threadIdx.x]);
}

void multiReduce64(int *keys, int *values, int *hist, int num_elements)
{
	cudaMemset(hist, 0, SHARED_HIST_SIZE * sizeof(int));
	computePartMRKernel<<<128, SHARED_HIST_SIZE>>>(keys, values, hist, num_elements / 128);
	cudaThreadSynchronize();
	printf("error after kernel: %i\n", cudaGetLastError());
}

int main()
{
printf("%i %i \n", NUMS_PER_GO, 96 / NUMS_PER_GO);
int num_elements = 1 << EXP; 

int *keys = (int*) malloc(num_elements * sizeof(int));
int *values = (int*) malloc(num_elements * sizeof(int));
int *result = (int*) malloc(SHARED_HIST_SIZE * sizeof(int));
int *result_cpu = (int*) malloc(SHARED_HIST_SIZE * sizeof(int));

for (int i = 0; i < num_elements; i++)
{
	keys[i] = 15;
//	keys[i] = rand() % SHARED_HIST_SIZE;
	values[i] = rand() % 10;
}

int *d_keys, *d_values, *d_result;
cudaMalloc((void**)&d_keys, num_elements * sizeof(int));
cudaMalloc((void**)&d_values, num_elements * sizeof(int));
cudaMalloc((void**)&d_result, SHARED_HIST_SIZE * sizeof(int));

cudaMemcpy(d_keys, keys, num_elements * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_values, values, num_elements * sizeof(int), cudaMemcpyHostToDevice);

timeval before, after, beforec, afterc;
gettimeofday(&before, NULL);
multiReduce64(d_keys, d_values, d_result, num_elements);
gettimeofday(&after, NULL);

cudaMemcpy(result, d_result, SHARED_HIST_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

gettimeofday(&beforec, NULL);
multiReduceCpu(keys, values, result_cpu, num_elements);
gettimeofday(&afterc, NULL);

printf("checking results...\n");
for (int i = 0; i < SHARED_HIST_SIZE; i++)
{
	if (result[i] != result_cpu[i])
		printf("i sense a discrepancy in bucket %i! %u %u\n", i, result[i], result_cpu[i]);
}

printf("cpu gpu ns/int %f %f \n", ((afterc.tv_sec - beforec.tv_sec) * 1e9 + (afterc.tv_usec - beforec.tv_usec) * 1e3) / num_elements, ((after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3) / num_elements);

}
