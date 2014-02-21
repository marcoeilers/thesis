#include <sys/time.h>
#include "nugteren/1_warp_private_histogramming/bitmap.cu"
#include "demoTRISH/histogram_common.h"


#define BINS 256
#define NO_RUNS 100

void histogramCpu(unsigned char *data, unsigned int *result, int num_elements);
void benchmark(unsigned char *data, unsigned int *result, unsigned int *result_cpu, int num_elements);
void divide (int size, int *width, int *height);


int main()
{
const int exponent = 28;
printf("input: 2^%i bytes (%i)\n", exponent, 1 << exponent);
int num_elements = 1 << exponent;

int width, height;

divide(num_elements, &width, &height);

unsigned int *result = (unsigned int*) malloc(BINS * sizeof(unsigned int));
unsigned int *result_cpu = (unsigned int*) malloc(BINS * sizeof(unsigned int));

// random data
printf("random data\n");

unsigned char *data = (unsigned char*) malloc(num_elements * sizeof(unsigned char));
for (int i = 0; i < num_elements; i++)
	data[i] = rand() % BINS;

benchmark(data, result, result_cpu, num_elements);
free(data);

// image 1
printf("image 1\n");

data = LoadBMPCustomDimensions(width, height, "image1.bmp");
benchmark(data, result, result_cpu, num_elements);
free(data);


// image 2
printf("image 2\n");

data = LoadBMPCustomDimensions(width, height, "image2.bmp");
benchmark(data, result, result_cpu, num_elements);
free(data);


// degenerate data
printf("degenerate data\n");
unsigned char index = rand() % BINS;
data = (unsigned char*) malloc(num_elements * sizeof(unsigned char));
for (int i = 0; i < num_elements; i++)
        data[i] = index;

benchmark(data, result, result_cpu, num_elements);
free(data);

}

void histogramCpu(unsigned char *data, unsigned int *result, int num_elements)
{
for (int i = 0; i < BINS; i++)
	result[i] = 0;
for (int i = 0; i < num_elements; i++)
	result[data[i]]++;
}

void benchmark(unsigned char *data, unsigned int *result, unsigned int *result_cpu, int num_elements)
{
unsigned int *d_result;
unsigned char *d_data;

cudaMalloc((void**) &d_result, BINS * sizeof(unsigned int));
cudaMalloc((void**) &d_data, num_elements);

cudaMemcpy(d_data, data, num_elements, cudaMemcpyHostToDevice);

timeval before, after;

initTrish256();
for (int i = -1; i < NO_RUNS; i++)
{
	if (i == 0)
	{
		cudaThreadSynchronize();
		gettimeofday(&before, NULL);
	}
//	initTrish256();
	histogramTrish256(d_result, d_data, num_elements);
//	closeTrish256();
}
cudaThreadSynchronize();
gettimeofday(&after, NULL);

cudaMemcpy(result, d_result, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
closeTrish256();
histogramCpu(data, result_cpu, num_elements);

bool correct = 1;
for (int i = 0; i < BINS; i++)
{
	if (result[i] != result_cpu[i])
	{
		printf("discrepancy: %i %u %u\n", i, result[i], result_cpu[i]);
		correct = 0;
	}
}

if (correct)
{
	printf("correct result!\n");
	float nsTotal = (after.tv_sec - before.tv_sec) * 1e9 + (after.tv_usec - before.tv_usec) * 1e3;
	printf("total %f ns/byte %f\n", nsTotal, (nsTotal / NO_RUNS) / num_elements);
}else{
	printf("incorrect result!\n");
}
cudaFree(d_result);
cudaFree(d_data);

}

void divide (int size, int *width, int *height)
{
*width = 1;
*height = 1;
int steps = 0;
while (size > 1)
{
	if (steps % 2 == 0)
	{
		size = size / 2;
		*width = *width * 2;
	}else{
		size = size / 2;
		*height = *height * 2;
	}
	steps++;
}
}
