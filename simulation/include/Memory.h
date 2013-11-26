#ifndef MEMORY_H
#define MEMORY_H


class Memory
{
    public:
//        Memory() {}
        virtual int getAddress(int*) = 0;
        virtual int fetchAddress(int*) = 0;
        virtual void putAddress(int*, int) = 0;
        virtual void storeAddress(int*, int) = 0;
        virtual void copy(int* from, int* to)
        {
            storeAddress(to, fetchAddress(from));
        }
    protected:
    private:
};

#endif // MEMORY_H
