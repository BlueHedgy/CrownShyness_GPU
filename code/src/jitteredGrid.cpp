#include "jitteredGrid.h"
#include <random>

using namespace LavaCake;


/*
    GenerateGrid:   randomize points line by line through an
                    abstractly subdivided grid
                    Currently: 1 point per cell per grid
*/

Grid2D generateGrid(u_int16_t subdivision, int seed){

    Grid2D grid;

    srand(seed);

    for(u_int16_t  j = 0; j < subdivision ; j++ ){
        std::vector<vec2f> line;
        for(u_int16_t  i = 0; i < subdivision ; i++ ){
            vec2f point;
            point [0] =  ((float)rand() / RAND_MAX + float(i)) / float(subdivision);
            point [1] =  ((float)rand() / RAND_MAX + float(j)) / float(subdivision);
            line.push_back(point);
        }
        grid.points.push_back(line);
    }

    return grid;
    
}

/* 
    Pseudocode for generateGrid(subdivision, seed):
        Grid2D grid;

        for (i=0; i < subdiv; i++){
            vector<2f> line;
            for (j=0; j < subdiv; j++){
                vector<vec2f> roots;
                for (t=0; t < maxTreePerCell; t++){
                    // generate x and y of points
                    // append point to roots
                }
                // append roots to line
            }
            // append line to grid.points   
            // gonna have to change the grid.points structure
        }
*/

vec2u closestPoint(const Grid2D& grid, const vec2f& point){

    vec2i cell({int(point[0] * grid.points[0].size() ),int(point[1] * grid.points.size())});
    vec2u closestPoint;
    float mindistsqrd = 10.0f;

    for(int j = cell[1]-1; j <= cell[1] +1 ; j++){
        for(int i = cell[0]-1; i <=cell[0]+1 ; i++){
            if( i >= 0 &&  i < grid.points[0].size()
                &&  j >= 0 && j < grid.points.size())
            {
                float distsqrd  = dot(point - grid.points[j][i],  point - grid.points[j][i]);

                if(distsqrd <mindistsqrd){
                    mindistsqrd = distsqrd;
                    closestPoint[0] = i;
                    closestPoint[1] = j;
                }
            }
        }
    }

    return closestPoint;

}