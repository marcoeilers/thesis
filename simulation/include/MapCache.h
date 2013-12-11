#ifndef MAPCACHE_H
#define MAPCACHE_H

#include <iostream>
#include <Memory.h>
#include <deque>
#include <map>

#define PREFETCH_LIMIT 32

#define ulong unsigned long


class MapCache : public Memory
{
    public:
        MapCache(Memory*, ulong*, int, int, int);
        virtual ~MapCache();
        virtual int getAddress(int*);
        virtual int fetchAddress(int*);
        virtual void putAddress(int*, int);
        virtual void storeAddress(int*, int);
        float missPercentage();
        void resetStatistics();
    protected:
    private:
        Memory* nextLevel;
        ulong* costVar;
        int cost;
        int size;
        int lines;
        int lineSize;
        std::map<std::size_t, std::size_t> entries;
        std::deque<std::size_t> order;
        void fetchToCache(int*);
        bool contains(int*);
        void pushToFront(int*);
        int accesses;
        int misses;
};

#endif // MAPCACHE_H
