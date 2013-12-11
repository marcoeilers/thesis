#ifndef RAM_H
#define RAM_H

#include <Memory.h>
#include <inttypes.h>

#define ulong uint64_t


class RAM : public Memory
{
    public:
        RAM(ulong*, int);
        virtual ~RAM();
        virtual int getAddress(int*);
        virtual int fetchAddress(int*);
        virtual void putAddress(int*, int);
        virtual void storeAddress(int*, int);
    protected:
    private:
        int cost;
        ulong* costVar;
};

#endif // RAM_H
