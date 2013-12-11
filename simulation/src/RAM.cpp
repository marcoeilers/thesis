#include "RAM.h"

RAM::RAM(ulong* cv, int c)
{
    costVar = cv;
    cost = c;
}

RAM::~RAM()
{
    //dtor
}

int RAM::getAddress(int* address)
{
    return *address;
}

int RAM::fetchAddress(int* address)
{
    (*costVar) += cost;
    return getAddress(address);
}
void RAM::putAddress(int* address, int value)
{
    *address = value;
}

void RAM::storeAddress(int* address, int value)
{
    (*costVar) += cost;
    putAddress(address, value);
}
