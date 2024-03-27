#include <iostream>
#include <fstream> 
#include <math.h>      

#include <LavaCake/Math/basics.h>
#include <vector>

using namespace LavaCake;

#ifndef DATA_STRUCTURES
#define DATA_STRUCTURES

struct Coord{
    uint32_t gridIndex;     // the layer which the point belong to
    vec2u coord;            // the cell of said layer
    u_int16_t pointIndex;   // the index of the point in the cell
    float weight; 
};

struct Edge{
    Coord c1, c2;
};

// struct points{
//     LavaCake::vec3f points;
// };

struct Cell{
    std::vector<LavaCake::vec2f> points;
    std::vector<float> points_weight;
};

struct Grid2D{
    std::vector<std::vector<Cell>> cells;
    std::vector<std::vector<int>> pointsCount; // keep track of number of points in grids and their cells

};

#endif

