#include "AssocCache.h"

template<uint LINES, uint ASSOC>
AssocCache<LINES, ASSOC>::AssocCache(Memory* next, ulong* cv, int cst, int ls, int s)
{
    nextLevel = next;
    costVar = cv;
    cost = cst;
    size = s;
    lines = LINES;
    lineSize = s / ls;
}

template<uint LINES, uint ASSOC>
AssocCache<LINES, ASSOC>::~AssocCache()
{
    //dtor
}

template<uint LINES, uint ASSOC>
int AssocCache<LINES, ASSOC>::getAddress(int* address)
{
    fetchToCache(address);
    return nextLevel->getAddress(address);
}

template<uint LINES, uint ASSOC>
int AssocCache<LINES, ASSOC>::fetchAddress(int* address)
{
    accesses++;
    if (contains(address))
    {
        pushToFront(address);
        (*costVar) += cost;
        return nextLevel->getAddress(address);
    }
    else
    {
        misses++;
        fetchToCache(address);
        return nextLevel->fetchAddress(address);
    }
}

template<uint LINES, uint ASSOC>
void AssocCache<LINES, ASSOC>::putAddress(int* address, int value)
{
    fetchToCache(address);
    nextLevel->putAddress(address, value);
}

template<uint LINES, uint ASSOC>
void AssocCache<LINES, ASSOC>::storeAddress(int* address, int value)
{
    accesses++;
    if (contains(address))
    {
        pushToFront(address);
        (*costVar) += cost;
        nextLevel->putAddress(address, value);
    }
    else
    {
        misses++;
        fetchToCache(address);
        nextLevel->storeAddress(address, value);
    }
}

template<uint LINES, uint ASSOC>
void AssocCache<LINES, ASSOC>::pushToFront(int* address)
{
    std::size_t add = (std::size_t) address;
    std::size_t start = (add / lineSize) * lineSize;

    int index = hash(start);
    int entryIndex = -1;
    std::size_t last = 0;
    for (int i = 0; i < ASSOC; i++)
    {
        if (entries[index][i] == start)
        {
            accessed[index] |= 1 << i;
            entryIndex = i;
            last = lastAcc[index][i];
            lastAcc[index][i] = add;
        }
    }
    if (accessed[index] == ALL_ONES)
        accessed[index] = 1 << entryIndex;

    if (add > last && add > (start + lineSize - PREFETCH_LIMIT))
    {
        if (! contains((int*) (start + lineSize)))
            fetchToCache((int*) (start + lineSize));
    }
}

template<uint LINES, uint ASSOC>
void AssocCache<LINES, ASSOC>::fetchToCache(int* address)
{
    std::size_t add = (std::size_t) address;
    std::size_t start = (add / lineSize) * lineSize;

    int index = hash(start);

    insert(add, index, start);
}

template<uint LINES, uint ASSOC>
void AssocCache<LINES, ASSOC>::insert(std::size_t add, int index, std::size_t start)
{
    int insertIndex = -1;
    for (int i = 0; i < ASSOC; i++)
    {
        if (!(accessed[index] & 1 << i))
        {
            entries[index][i] = start;
            accessed[index] |= 1 << i;
            lastAcc[index][i] = add;
            insertIndex = i;
            break;
        }
    }
    if (accessed[index] = ALL_ONES)
        accessed[index] = 1 << insertIndex;
}

template<uint LINES, uint ASSOC>
bool AssocCache<LINES, ASSOC>::contains(int* address)
{
    std::size_t add = (std::size_t) address;
    std::size_t start = (add / lineSize) * lineSize;
    int index = hash(start);
    for (int i = 0; i < ASSOC; i++)
    {
        if (entries[index][i] == start)
            return true;
    }
    return false;
}

template<uint LINES, uint ASSOC>
int AssocCache<LINES, ASSOC>::hash(std::size_t address)
{
    return (address % PRIME) % (LINES / ASSOC);//((address % PRIME) ^ (address > 1)) % (LINES / ASSOC);
}

template<uint LINES, uint ASSOC>
float AssocCache<LINES, ASSOC>::missPercentage()
{
    return ((float) misses) / ((float) accesses);
}

template<uint LINES, uint ASSOC>
void AssocCache<LINES, ASSOC>::resetStatistics()
{
    misses = 0;
    accesses = 0;
}
