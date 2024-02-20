#include "jitteredGrid.h"
#include <random>

using namespace LavaCake;


/*
    GenerateGrid:   randomize points line by line through an
                    abstractly subdivided grid
                    Currently: 1 point per cell per grid
*/
u_int16_t maxTreePerCell = 4;

Grid2D generateGrid(u_int16_t subdivision, int seed){

    Grid2D grid;

    srand(seed);

    for(u_int16_t  j = 0; j < subdivision ; j++ ){
        std::vector<Cell> currentCellRow;
        for(u_int16_t  i = 0; i < subdivision ; i++ ){

            Cell currentCell;
            // generate maxTreePerCell amount of points for each cell
            for (u_int16_t c = 0; c < maxTreePerCell; c++){
                vec2f point;
                point [0] =  ((float)rand() / RAND_MAX + float(i)) / float(subdivision);
                point [1] =  ((float)rand() / RAND_MAX + float(j)) / float(subdivision);
                currentCell.points.push_back(point);
            }
            currentCellRow.push_back(currentCell);
        }
        grid.cells.push_back(currentCellRow);
    }

    return grid;
    
}


Coord getClosestPoint(const Grid2D& grid, const vec2f& point, const  uint32_t gridLayer){

    // get the corresponding cell in the lower level projected from the current point
    vec2i cell({int(point[0] * grid.cells[0].size() ),int(point[1] * grid.cells.size())});
    Coord closestPoint;
    float mindistsqrd = 10.0f;

    for(int j = cell[1]-1; j <= cell[1] +1 ; j++){
        for(int i = cell[0]-1; i <=cell[0]+1 ; i++){
            if( i >= 0 &&  i < grid.cells[0].size() &&  j >= 0 && j < grid.cells.size()){
                
                for (int p = 0; p < grid.cells[i][j].points.size(); p++){
                    float distsqrd  = dot(point - grid.cells[j][i].points[p],  point - grid.cells[j][i].points[p]);

                    if(distsqrd <mindistsqrd){
                        mindistsqrd = distsqrd;
                        // closestPoint[0][0] = i;
                        // closestPoint[0][1] = j;
                        // closestPoint[0][2] = p;
                        closestPoint.gridIndex = gridLayer;
                        closestPoint.coord[0] = i;
                        closestPoint.coord[1] = j;
                        closestPoint.pointIndex = p;

                    }
                }
            }
        }
    }

    return closestPoint;

}