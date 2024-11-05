#pragma once

template <typename T>

class Tensor{
private:
    std::vector<T> data;
    std::vector<T> stride_;
    int64_t dims_;
    std::vector<T> shape_;
    int totalSize_;
    T type_;

public:
    Tensor(std::vector<T> shape, T dataType, int elements, std::vector<T> stride) : shape(shape_) , dataType(type_), elements(totalSize_), stride(stride_) {};
    
    // Functions  
    // adding
    T Tensor(T arr[][], T num)
    {
        for (i = 0; i < arr.size(); i++)
        {
            for (j = 0; j < arr.size(); j++)
            {
                arr[i][j] = arr[i][j] + num;
            }
        }
    } 
    // subtracting
    T Tensor(T arr[][], T num)
    {
        for (i = 0; i < arr.size(); i++)
        {
            for (j = 0; j < arr.size(); j++)
            {
                arr[i][j] = arr[i][j] -  num;
            }
        }
    } 
    // scalar multpliaction
    T Tensor(T arr[][], T num)
    {
        for (i = 0; i < arr.size(); i++)
        {
            for (j = 0; j < arr.size(); j++)
            {
                arr[i][j] = arr[i][j] *  num;
            }
        }
    }  
    // scalar division
    T Tensor(T arr[][], T num)
    {
        for (i = 0; i < arr.size(); i++)
        {
            for (j = 0; j < arr.size(); j++)
            {
                arr[i][j] = (arr[i][j]/ num);
            }
        }
    }
    // requires matrix multplication
    T matmul(T arr1[][], T arr2[][])
    {
        for (i = 0; i < arr.size(); i++)
        {
            for (j = 0; j < arr.size(); j++)
            {
                arr1[i][j] = arr1[i][j] * arr2[i][j];
            }
        }
    }
    // element wise multiplication
};