#pragma once

#include "array.h"

template <typename T = float>
class Tensor {
 private:
    Array<T>* data;
    Array<T>* grad;
    vector<int> shape;

 public:
    Tensor(vector<int> shape) : shape(shape) {
        data = new Array<T>(shape);
        grad = new Array<T>(shape);
    }

    ~Tensor() {
        delete data;
        delete grad;
    }
    
    // Functions 

    //no need for tensor zero. the array is inhertly 0.
    Tensor random_tensor(float lower, float upper) {
        data->randomize(lower, upper);
        grad->randomize(lower, upper);
    }

    // // adding
    // T Tensor(T arr[][], T num)
    // {
    //     for (i = 0; i < arr.size(); i++)
    //     {
    //         for (j = 0; j < arr.size(); j++)
    //         {
    //             arr[i][j] = arr[i][j] + num;
    //         }
    //     }
    // } 
    // // subtracting
    // T Tensor(T arr[][], T num)
    // {
    //     for (i = 0; i < arr.size(); i++)
    //     {
    //         for (j = 0; j < arr.size(); j++)
    //         {
    //             arr[i][j] = arr[i][j] -  num;
    //         }
    //     }
    // } 
    // // scalar multpliaction
    // T Tensor(T arr[][], T num)
    // {
    //     for (i = 0; i < arr.size(); i++)
    //     {
    //         for (j = 0; j < arr.size(); j++)
    //         {
    //             arr[i][j] = arr[i][j] *  num;
    //         }
    //     }
    // }  
    // // scalar division
    // T Tensor(T arr[][], T num)
    // {
    //     for (i = 0; i < arr.size(); i++)
    //     {
    //         for (j = 0; j < arr.size(); j++)
    //         {
    //             arr[i][j] = (arr[i][j]/ num);
    //         }
    //     }
    // }
    // // requires matrix multplication
    // T matmul(T arr1[][], T arr2[][])
    // {
    //     for (i = 0; i < arr.size(); i++)
    //     {
    //         for (j = 0; j < arr.size(); j++)
    //         {
    //             arr1[i][j] = arr1[i][j] * arr2[i][j];
    //         }
    //     }
    // }
    // element wise multiplication
};
