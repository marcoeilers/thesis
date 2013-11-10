#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <limits.h>

#define NO_BITS 8
#define RUNS 4
#define NO_REPS 10

#define ONE_BITS ((1 << NO_BITS) - 1)
#define ByteOf(x) (((x) >> bitsOffset) & ONE_BITS)
#define BteOf(x, y) (((x) >> y) & ONE_BITS)

int counts[RUNS][1 << NO_BITS];

static void getCounts(int size, int *values)
{
  int *cp = &counts[0][0];
  for (int i = RUNS * (1 << NO_BITS); i > 0; --i, ++cp)
    *cp = 0;
  
  int *sp = values;
  for (int i = 1 << NO_BITS; i > 0; --i, ++sp){
    for (int j = 0; j < RUNS; j++){
      cp = (&counts[j][0]) + BteOf(*sp, j);
      ++(*cp);
    }
  }
      
}


static void radix (int run, short bitsOffset, int size, int *values, int *result)
{
	int *cp, *sp;

	// scan 
	int sum = 0;
	cp = counts[run];
	for (int i = (1 << NO_BITS); i > 0; --i, ++cp) {
		int cur = *cp;
		*cp = sum;
		sum += cur;
	}

	sp = values;
	for (int i = size; i > 0; --i, ++sp) {
		int cur = *sp;
		cp = counts[run] + ByteOf (cur);
		result[*cp] = cur;
//                printf("writing %i to %i.\n", s, *cp);
		++(*cp);
	}
}

static void radix_sort (int *source, int *temp1, int *temp2, int size)
{
//        struct timeval before, after;
//        gettimeofday(&before, NULL);
        getCounts(size, source);
//        gettimeofday(&after, NULL);
//        float totalNs = 1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec);
//        printf("getcounts %f \n", totalNs);
        
        for (int i = 0; i < RUNS; i++){
//          gettimeofday(&before, NULL);
          radix(i, i * NO_BITS, size, i == 0 ? source : (i % 2 == 0 ? temp1 : temp2), i % 2 == 0 ? temp1 : temp2);
//          gettimeofday(&after, NULL);
//          totalNs = 1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec);
//          printf("run %i %f\n", i, totalNs);
        }

}

static void make_random (int *data, int size)
{
	for ( ; size > 0; --size, ++data)
		*data = rand () % INT_MAX;
}

static void check_order (int *data, int size)
{
	// only signal errors if any (should not be)
	for (--size ; size > 0; --size, ++data)
		assert (data[0] <= data[1]);
}

int cmpfunc (const void * a, const void * b)
{
   return ( *(int*)a - *(int*)b );
}


static void test_radix (int size)
{
	unsigned i;
        int *data = calloc (size, sizeof (int));
        int *temp1 = calloc (size, sizeof(int));
        int *temp2 = calloc (size, sizeof(int));
        for (int i = 0; i < size; i++)
        {
          data[i] = i;
          temp1[i] = 13;
          temp2[i] = 25;
        }

        struct timeval before, after;

	make_random (data, size);
        gettimeofday(&before, NULL);
        for (int i = 0; i < NO_REPS; i++)
	  radix_sort (data, temp1, temp2, size);
        gettimeofday(&after, NULL);
        float totalNs = (1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec)) / NO_REPS;
        printf("%i\t%f\n", size, totalNs / size);
//	check_order (RUNS % 2 == 0 ? temp2 : temp1, size);

/*	make_random (data, size);
        gettimeofday(&before, NULL);
        qsort(data, size, sizeof(int), cmpfunc);
        gettimeofday(&after, NULL);
        totalNs = 1e9 * (after.tv_sec - before.tv_sec) + 1e3 * (after.tv_usec - before.tv_usec);
        printf("qsort ns/int %f \n", totalNs / size);*/
        
	free (data);
        free (temp1);
        free (temp2);
}



int main (void)
{
        for (int i = 23; i < 29; i++)
	  test_radix (1 << i);
	return 0;
}
