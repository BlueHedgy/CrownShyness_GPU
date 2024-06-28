#include <iostream>
#include <fstream> 
#include <math.h>

#include <numeric>
#include <string>
#include <set>
#include <map>


#include <LavaCake/Math/basics.h>
#include <vector>

using namespace LavaCake;

#ifndef DATA_STRUCTURES
#define DATA_STRUCTURES

enum TREE_TYPE{
    Columnar,
    Pyramidal,
    Oval,
    Rounded,
    Spreading,
    Vase,
    Weeping,
    SIZE=7
};

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

// Data structures after trees have been fully determined

struct Point {
    int grid_index;
    vec3f position;
    // vec3f children_center;
    int parent = -1;
    std::vector<int> children;
};

// inline bool operator<(const Point &a, const Point &b){
//     return a.index < b.index;
// }

struct Branch{
    int i2, i1;
    int k2, k1;
};

struct Tree{
    int ID;

    std::map<int, Point> points;

    std::vector<Branch> branches;

    int numBranches;
    TREE_TYPE type;

    vec3f center = vec3f({0.0f, 0.0f, 0.0f});
    float SHRINK_FACTOR = 0.0f;
};


#endif

