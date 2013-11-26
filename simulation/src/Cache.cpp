#include "Cache.h"

Cache::Cache(Memory* next, int* cv, int cst, int ls, int s)
{
    nextLevel = next;
    costVar = cv;
    cost = cst;
    size = s;
    lines = ls;
    lineSize = s / ls;
}

Cache::~Cache()
{
    //dtor
}

int Cache::getAddress(int* address)
{
    return nextLevel->getAddress(address);
}

int Cache::fetchAddress(int* address)
{
    if (contains(address))
    {
        pushToFront(address);
        (*costVar) += cost;
        return nextLevel->getAddress(address);
    }
    else
    {
        fetchToCache(address);
        return nextLevel->fetchAddress(address);
    }
}

void Cache::putAddress(int* address, int value)
{
    nextLevel->putAddress(address, value);
}

void Cache::storeAddress(int* address, int value)
{
    if (contains(address))
    {
        pushToFront(address);
        (*costVar) += cost;
        nextLevel->putAddress(address, value);
    }
    else
    {
        fetchToCache(address);
        nextLevel->storeAddress(address, value);
    }
}

void Cache::pushToFront(int* address)
{
    std::size_t add = (std::size_t) address;
    CacheEntry temp;
    std::deque<CacheEntry>::iterator it;
    for (it = entries.begin(); it != entries.end(); it++)
    {
        bool first = it->start <= add;
        std::size_t end = it->start + it->size;
        bool second = end > add;
        if (first && second)
        {
            temp = *it;
            entries.erase(it);
            entries.push_front(temp);

            // prefetch if necessary
            if (add > temp.lastAccess && add > (temp.start + temp.size - PREFETCH_LIMIT))
                fetchToCache((int*) (temp.start + temp.size));
        }
        break;
    }
}

void Cache::fetchToCache(int* address)
{
    std::size_t add = (std::size_t) address;
    std::size_t start = (add / lineSize) * lineSize;
    CacheEntry temp;
    temp.start = start;
    temp.size = lineSize;
    temp.lastAccess = add;
    entries.push_front(temp);

    if (entries.size() > lines)
        entries.pop_back();
}

bool Cache::contains(int* address)
{
    std::size_t add = (std::size_t) address;

    std::size_t segStart = (add / lineSize) * lineSize;
    std::deque<CacheEntry>::iterator it;
    for (it = entries.begin(); it < entries.end(); it++)
    {
        if (it->start == segStart)
            return true;
    }
    return false;
}
