#ifndef RAM_H
#define RAM_H

#include <Memory.h>


class RAM : public Memory
{
    public:
        RAM(int*, int);
        virtual ~RAM();
        virtual int getAddress(int*);
        virtual int fetchAddress(int*);
        virtual void putAddress(int*, int);
        virtual void storeAddress(int*, int);
    protected:
    private:
        int cost;
        int* costVar;
};

#endif // RAM_H
