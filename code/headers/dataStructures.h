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
    uint16_t pointIndex;   // the index of the point in the cell
    float weight; 
    int tree_index;
};

struct Edge{
    Coord c1, c2;
};

struct point_Info{
    float points_weight;
    int tree_index;
    int global_point_index;
};

struct Cell{
    std::vector<LavaCake::vec3f> points;
    std::vector<point_Info> pointsInfo;
};

struct Grid2D{
    std::vector<std::vector<Cell>> cells;
    std::vector<std::vector<int>> pointsCount; // keep track of number of points in grids and their cells

};

struct Tree{
    int ID;
    std::vector<Edge> branches;
    int numEdges;
    
};


#endif

