#include <iostream>
#include <fstream> 
#include <math.h>

#include <numeric>
#include <string>
#include <set>
#include <map>


#include <LavaCake/Math/basics.h>
#include "dataStructures.h"
#include <vector>

using namespace LavaCake;

#ifndef DATA_STRUCTURES_GPU
#define DATA_STRUCTURES_GPU


struct Cell_GPU{

    float * points;

    __host__ __device__
    Cell_GPU() : points(nullptr) {} 

    __host__ __device__
    Cell_GPU(const int size){
        // cudaMalloc(&points, size * 3 * sizeof(float));
        points = new float[size * 3];
    }

    __host__ __device__
    ~Cell_GPU(){}   // Placeholder destructor
};

struct Grid2D_GPU{

    Cell_GPU *cells; 
    int *pointsCount;

    __device__
    Grid2D_GPU(const int size){
        cudaMalloc(&cells, size * sizeof(Cell_GPU));
        cudaMalloc(&pointsCount, size * sizeof(int));
    }

    __device__
    ~Grid2D_GPU(){}   // Placeholder destructor

};


#endif

