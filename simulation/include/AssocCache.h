#ifndef ASSOCCACHE_H
#define ASSOCCACHE_H

#include <iostream>
#include <Memory.h>
#include <deque>
#include <map>
#include <inttypes.h>

#define PREFETCH_LIMIT 32
#define PRIME 181081
#define ALL_ONES ((1 << ASSOC) - 1)

#ifndef ulong
#define ulong uint64_t
#endif

#ifndef uint
#define uint unsigned int
#endif


template<uint LINES, uint ASSOC>
class AssocCache : public Memory
{
    public:
        AssocCache(Memory*, ulong*, int, int, int);
        virtual ~AssocCache();
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
        std::size_t entries[LINES / ASSOC][ASSOC];
        std::size_t lastAcc[LINES/ASSOC][ASSOC];
        int accessed[LINES / ASSOC];
        int hash(std::size_t);
        void fetchToCache(int*);
        bool contains(int*);
        void pushToFront(int*);
        void insert(std::size_t, int, std::size_t);
        int accesses;
        int misses;
};

#include "AssocCache.hpp"

#endif // ASSOCCACHE_H
