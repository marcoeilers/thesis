#ifndef CACHE_H
#define CACHE_H

#include <Memory.h>
#include <deque>
#include <iostream>

#define PREFETCH_LIMIT 32

struct CacheEntry
{
    std::size_t start;
    int size;
    std::size_t lastAccess;
};

class Cache : public Memory
{
    public:
        Cache(Memory*, int*, int, int, int);
        virtual ~Cache();
        virtual int getAddress(int*);
        virtual int fetchAddress(int*);
        virtual void putAddress(int*, int);
        virtual void storeAddress(int*, int);
    protected:
    private:
        Memory* nextLevel;
        int* costVar;
        int cost;
        int size;
        int lines;
        int lineSize;
        std::deque<CacheEntry> entries;
        void fetchToCache(int*);
        bool contains(int*);
        void pushToFront(int*);
};

#endif // CACHE_H
