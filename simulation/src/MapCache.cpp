#include "MapCache.h"

MapCache::MapCache(Memory* next, ulong* cv, int cst, int ls, int s)
{
    nextLevel = next;
    costVar = cv;
    cost = cst;
    size = s;
    lines = ls;
    lineSize = s / ls;
}

MapCache::~MapCache()
{
    //dtor
}

int MapCache::getAddress(int* address)
{
    fetchToCache(address);
    return nextLevel->getAddress(address);
}

int MapCache::fetchAddress(int* address)
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

void MapCache::putAddress(int* address, int value)
{
    fetchToCache(address);
    nextLevel->putAddress(address, value);
}

void MapCache::storeAddress(int* address, int value)
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

void MapCache::pushToFront(int* address)
{
    std::size_t add = (std::size_t) address;
    std::size_t start = (add / lineSize) * lineSize;

    if (order.front() != start){
    std::deque<std::size_t>::iterator it;
    for (it = order.begin(); it != order.end(); it++)
    {
        if (*it == start)
        {
            order.erase(it);
            break;
        }
    }

    order.push_front(start);
    }

    std::size_t lastAcc = entries[start];
    if (add > lastAcc && add > (start + lineSize - PREFETCH_LIMIT))
        fetchToCache((int*) (start + lineSize));
}

void MapCache::fetchToCache(int* address)
{
    std::size_t add = (std::size_t) address;
    std::size_t start = (add / lineSize) * lineSize;

    entries.insert(std::pair<std::size_t, std::size_t>(start, add));
    order.push_front(start);

    if (entries.size() > lines)
    {
        std::size_t last = order.back();
        order.pop_back();
        entries.erase(last);
    }
}

bool MapCache::contains(int* address)
{
    std::size_t add = (std::size_t) address;

    std::size_t segStart = (add / lineSize) * lineSize;
    return (entries.find(segStart) != entries.end());
}

float MapCache::missPercentage()
{
    return ((float) misses) / ((float) accesses);
}

void MapCache::resetStatistics()
{
    misses = 0;
    accesses = 0;
}
