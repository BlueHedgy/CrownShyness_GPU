#include "jitteredGrid.h"
#include <random>

using namespace LavaCake;

std::vector<LavaCake::vec2f> randomizeDenseCenter(int dense_region_count, int init_subdiv){
    std::vector<LavaCake::vec2f> dense_centers;
    for (int i=0; i< dense_region_count; i++){
        float x = rand() % init_subdiv + ((float)rand() / RAND_MAX + float(i));
        float y = rand() % init_subdiv + ((float)rand() / RAND_MAX + float(i));
        dense_centers.push_back(vec2f({x,y}));
    }

    return dense_centers;
}

std::vector<std::vector<float>> density_map (int dense_region_count, int subdiv){

    std::vector<LavaCake::vec2i> dense_centers;
    std::vector<std::vector<float>> map;

    // Pick randomly dense_region_count cells in the current grid as density center
    for (int i = 0; i < dense_region_count; i++){
        int x = rand() % subdiv;
        int y = rand() % subdiv;
        dense_centers.push_back(vec2i({x,y}));
    }

    // generate a weighted map to be used with generate grid

    for (int j = 0; j < subdiv; j++){
        std::vector<float> currentRow;

        for(int i = 0; i < subdiv; i++){ 
            float minDistSqr = 10000.0f;
            for (auto c: dense_centers){
                float distSqr = dot(c - vec2i({i, j}), c - vec2i({i, j}));
                if (distSqr < minDistSqr){
                    minDistSqr = distSqr;
                }
            }
            currentRow.push_back(1.0 - sqrt(minDistSqr)/(subdiv * sqrt(2)));
        }
        map.push_back(currentRow);
    }
    
    return map;
    
}

/*
    GenerateGrid:   randomize points line by line through an
                    abstractly subdivided grid
                    Currently: 1 point per cell per grid
*/
u_int16_t maxPointsPerCell = 6;

Grid2D generateGrid(u_int16_t subdivision, int seed){
//TODO : fix this
    Grid2D grid;
    srand(seed);
    std::vector<std::vector<float>> weight_map = density_map(1, subdivision);


    float init_subdiv = 2;
    for(u_int16_t  j = 0; j < subdivision ; j++ ){
        std::vector<Cell> currentCellRow;
        std::vector<int> currentCellPointsCount;

        for(u_int16_t  i = 0; i < subdivision ; i++ ){

            int pointCount = int(float(maxPointsPerCell) * sin(float(j)/float(subdivision) *2.0*3.14159265 * 10.0));
            if (pointCount < 1) pointCount = 1;
            // for (auto c: randomCenters){
            //     vec2i center;
            //     center[0] = ceil((c[0]/ init_subdiv) * subdivision);
            //     center[0] = ceil((c[0]/ init_subdiv) * subdivision);
            //     center[1] = 
            // }

            // int pointCount = 3;
            Cell currentCell;
            std::cout<<pointCount<<std::endl;
            for (u_int16_t c = 0; c < pointCount; c++){
                vec2f point;
                point [0] =  ((float)rand() / RAND_MAX + float(i)) / float(subdivision);
                point [1] =  ((float)rand() / RAND_MAX + float(j)) / float(subdivision);
                currentCell.points.push_back(point);
            }
            currentCellRow.push_back(currentCell);
            currentCellPointsCount.push_back(pointCount);
        }
        grid.cells.push_back(currentCellRow);
        grid.pointsCount.push_back(currentCellPointsCount);
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
            // make sure the cells that are being checked is within the boundary of the grids
            if( i >= 0 &&  i < grid.cells[0].size() &&  j >= 0 && j < grid.cells.size()){
                // iterating through the points in the cell
                for (int p = 0; p < grid.cells[j][i].points.size(); p++){

                    float distsqrd  = dot(point - grid.cells[j][i].points[p],  point - grid.cells[j][i].points[p]);

                    if(distsqrd <mindistsqrd){
                        mindistsqrd = distsqrd;

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