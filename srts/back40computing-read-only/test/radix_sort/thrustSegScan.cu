#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
//#include <thrust/execution_policy.h>
#include "valIndex.h"


int main()
{
int keys[10] = {0, 0, 0, 1, 1, 2, 3, 3, 3, 3};
valIndex vals[10];
for (int i = 0; i < 10; i++)
{
	vals[i].value = rand() % 20;
	vals[i].index = i;
}
valIndex init;
init.value = 0;
init.index = 0;
thrust::equal_to<int> binary_pred;
valIndexAdd     binary_op;
thrust::inclusive_scan_by_key(keys, keys + 10, vals, vals, binary_pred, binary_op); // in-place scan
for (int i = 0; i < 10; i++)
{
printf("%i\t%i\n", vals[i].value, vals[i].index);
}
}
