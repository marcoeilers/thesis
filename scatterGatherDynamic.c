#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#define NO_REPS 10

#define SIMPLE_SCATTER
//#define SIMPLE_GATHER
#define HUGE

/*
$ echo 20 > /proc/sys/vm/nr_hugepages
$ sysctl -w vm.nr_hugepages=1024
$ echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
$ mount -t hugetlbfs nodev /mnt/huge
*/

#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#define FILE_NAME "/mnt/huge/hpage"
#define PROTECTION (PROT_READ | PROT_WRITE)

/* Only ia64 requires this */
#ifdef __ia64__
#define ADDR (void *)(0x8000000000000000UL)
#define FLAGS (MAP_SHARED | MAP_FIXED)
#else
#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_SHARED)
#endif

int _allocated_size;
int _fd;

void* lmalloc(int length)
{
  void *addr;

  _fd = open(FILE_NAME, O_CREAT | O_RDWR, 0755);
  if (_fd < 0) 
  {
    perror("Open failed");
    exit(1);
  }

  addr = mmap(ADDR, length, PROTECTION, FLAGS, _fd, 0);
  if (addr == MAP_FAILED) 
  {
    perror("mmap");
    unlink(FILE_NAME);
    exit(1);
  }
  
  _allocated_size = length;  
  return addr;
}

void lfree(void *addr)
{
  munmap(addr, _allocated_size);
  close(_fd);
  unlink(FILE_NAME);
}

void histogram(int *in, int *result, int length)
{
  int nextLabel = in[0];
  for (int i = 1; i < length; i++)
  {
    int current = nextLabel;
    nextLabel = in[i];
    result[current]++;
  }
  result[nextLabel]++;
}

void main(int argc, char *argv[])
{
  srand(time(NULL));
//  for (int i = 23; i < 29; i++){
  int size = 1 << atoi(argv[1]);


#ifdef HUGE 
  int *memory = (int*) lmalloc(size * sizeof(int) * 3);
#else
  int *memory = (int*) calloc(size * 3, sizeof(int));
#endif
  int *indices = memory;
  int *values = indices + size;
  int *result = values + size;
//  int *resultR = result + size;

  for (int i = 0; i < size; i++)
  {
    values[i] = i;
    result[i] = 15;
//    resultR[i] = 15;
    indices[i] = rand() % size;
  }

  struct timeval before, after;
  float totalNs;
#ifdef SIMPLE_SCATTER  
  gettimeofday(&before, NULL);
  for (int i = 0; i < NO_REPS; i++)
    simpleScatter(values, indices, result, size);
  gettimeofday(&after, NULL);
  totalNs = (1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec)) / NO_REPS;
  printf("%i\t%f\n", size, totalNs / size);
#endif
//  printResult(values, indices, resultR, size);
#ifdef SIMPLE_GATHER
  gettimeofday(&before, NULL);
  for (int i = 0; i < NO_REPS; i++)
    simpleGather(values, indices, result, size);
  gettimeofday(&after, NULL);
  totalNs = (1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec)) / NO_REPS;
  printf("%i\t%f\n", size, totalNs / size);
#endif
/*
  reInit(result, size);
  gettimeofday(&before, NULL);
  for (int i = 0; i < NO_REPS; i++)
    simpleScatter2(values, indices, result, size);
  gettimeofday(&after, NULL);
  totalNs = (1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec)) / NO_REPS;
  printf("simpleScatter2 %f ns/int\n", totalNs / size);
  checkEquality(result, resultR, size);

  reInit(result, size);
  gettimeofday(&before, NULL);
  for (int i = 0; i < NO_REPS; i++)
    simpleGather2(values, indices, result, size);
  gettimeofday(&after, NULL);
  totalNs = (1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec)) / NO_REPS;
  printf("simpleGather2 %f ns/int\n", totalNs / size);

  reInit(result, size);
  gettimeofday(&before, NULL);
  for (int i = 0; i < NO_REPS; i++)
    simpleGather3(values, indices, result, size);
  gettimeofday(&after, NULL);
  totalNs = (1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec)) / NO_REPS;
  printf("simpleGather3 %f ns/int\n", totalNs / size);

  for (int parts = 2; parts < 17; parts *= 2)
  {
    reInit(result, size);
    gettimeofday(&before, NULL);
    for (int i = 0; i < NO_REPS; i++)
      partScatter(values, indices, result, size, parts);
    gettimeofday(&after, NULL);
    totalNs = (1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec)) / NO_REPS;
    printf("partScatter %i %f ns/int\n", parts, totalNs / size);
    checkEquality(result, resultR, size);
//    printResult(values, indices, result, size);
  }
*/
#ifdef HUGE
  lfree(memory);
#else
  free(memory);
#endif
//  }
}

void simpleScatter(int *values, int *indices, int *result, int size)
{
  for (int i = 0; i < size; i++)
  {
    result[indices[i]] = values[i];
  }
}

void simpleScatter2(int *values, int *indices, int *result, int size)
{
  int nextIndex = indices[0];
  int nextValue = values[0];
  for (int i = 0; i < size - 1; i++)
  {
    int curIndex = nextIndex;
    nextIndex = indices[i+1];
    int curValue = nextValue;
    nextValue = values[i+1];
    result[curIndex] = curValue;
  }
  result[nextIndex] = nextValue;
}

void simpleGather(int *values, int *indices, int *result, int size)
{
  for (int i = 0; i < size; i++)
  {
    result[i] = values[indices[i]];
  }
}

void simpleGather2(int *values, int *indices, int *result, int size)
{
  int nextIndex = indices[0];
  for (int i = 0; i < size - 1; i++)
  {
    int curIndex = nextIndex;
    nextIndex = indices[i+1];
    result[i] = values[curIndex];
  }
  result[size - 1] = values[nextIndex];
}

void simpleGather3(int *values, int *indices, int *result, int size)
{
  int nextValue = values[indices[0]];
  int nextIndex = indices[1];
  for (int i = 0; i < size - 2; i++)
  {
    int curValue = nextValue;
    nextValue = values[nextIndex];
    nextIndex = indices[i + 2];
    result[i] = curValue;
  }
  result[size - 2] = nextValue;
  result[size - 1] = values[nextIndex];
}

void partScatter(int *values, int *indices, int *result, int size, int parts)
{
  int range = size / parts; 
  for (int i = 0; i < parts; i++)
  {
    int lowBound = i * range;
    int upBound = (i+1) * range;
    for (int j = 0; j < size; j++)
    {
      int curIndex = indices[j];
      if (curIndex >= lowBound && curIndex < upBound)
      {
        result[curIndex] = values[j];
      }
    } 
  }
}

void checkEquality(int *first, int *second, int size)
{
  int equal = 1;
  for (int i = 0; i < size; i++)
    equal = equal && (first[i] == second[i]);
  if (!equal)
  {
    printf("wrong result.\n");
  }
}

void printResult(int *values, int *indices, int *result, int size)
{
  for(int i = 0; i < size; i++)
    printf("%i\t%i\t%i\t%i\n", i, values[i], indices[i], result[i]);
}

void reInit(int *result, int size)
{
  for (int i = 0; i < size; i++)
    result[i] = 15;
}
