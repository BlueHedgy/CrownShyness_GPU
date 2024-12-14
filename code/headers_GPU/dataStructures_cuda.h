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

struct Coord{
    uint32_t gridIndex;         // the layer which the point belong to
    vec2u coord;                // the cell of said layer
    uint16_t pointIndex;        // the index of the point in the cell
    float weight;
    float strength;
    int tree_index;
};

struct Edge{
    Coord c1, c2;
};

struct point_Info{
    float points_weight;
    float strength;             // to be compared against gravity
    int tree_index;
    int global_point_index;
};

struct Cell{
    // std::vector<LavaCake::vec3f> points;
    // std::vector<point_Info> pointsInfo;

    LavaCake::vec3f * points;
    point_Info *pointsInfo;

    Cell(const int size){
        cudaMalloc(&points, size * sizeof(LavaCake::vec3f));
        cudaMalloc(&pointsInfo, size * sizeof(point_Info));
    }

    ~Cell(){}   // Placeholder destructor
};

struct Grid2D{
    // std::vector<std::vector<Cell>> cells;
    // std::vector<std::vector<int>> pointsCount; // keep track of number of points in grids and their cells

    Cell *cells; 
    int *pointsCount;
 
    __host__ __device__
    Grid2D(const int size){
        cudaMalloc(&cells, size * sizeof(Cell));
        cudaMalloc(&pointsCount, size * sizeof(int));
    }

    ~Grid2D(){}   // Placeholder destructor

};

// Data structures after trees have been fully determined

/*  Point:

    grid_index: the layer it was taken from
    position: 3d coordinates
    parent: index of the parent branching point
    children: list of indices of children point in EDGE PAIRS
    direction: direction of the previous branch curve's last segment
    prevNumSegments: parent branch number of segments
    prevIndices: parent branch's list of indices
    prevLength: parent EDGE's length
*/

struct Point {
    int grid_index;
    vec3f *position;
    int parent = -1;
    std::vector<int> children;
    vec3f direction;
    int prevNumSegments;
    int lastSegmentIndex;
    std::vector <int> prevIndices;  
    float strength;
    float prevLength;
};


struct Branch{
    int i2, i1;
    int k2, k1; // corresponding grid indices 
};

struct Tree{
    int ID;

    std::map<int, Point> points;

    std::vector<Branch> branches;

    std::vector<Branch> spline_Branches;

    int numBranches = 0;
    int numSplineBranches = 0;

    vec3f center = vec3f({0.0f, 0.0f, 0.0f});
    vec3f rootDirection = vec3f({0.0f, 0.0f, 0.0f});
    float SHRINK_FACTOR = 0.0f;
};


#endif

