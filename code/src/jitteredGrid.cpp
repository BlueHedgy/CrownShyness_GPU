#include "jitteredGrid.h"
#include <random>

using namespace LavaCake;


/*
    GenerateGrid:   randomize points line by line through an
                    abstractly subdivided grid
                    Currently: 1 point per cell per grid
*/
uint16_t maxPointsPerCell = 8;
int tree_index = -1;

Grid2D generateGrid(uint16_t subdivision, int seed, int gridLayer, std::string filename, int &point_index){

    Grid2D grid;
    srand(seed + 124534);

    std::vector<std::vector<float>> weight_map;

    if (!filename.empty()){
        weight_map = user_density_map(filename, subdivision);
    }
    
    for(uint16_t j = 0; j < subdivision ; j++){
        std::vector<Cell> currentCellRow;
        std::vector<int> currentCellPointsCount;

        for(uint16_t  i = 0; i < subdivision ; i++){

            int pointCount;

            if (!filename.empty()){
                pointCount = int(float(MAX_POINT_PER_CELL) * weight_map[j][i]);
            }
            else{
                pointCount = MAX_POINT_PER_CELL;
            }


            Cell currentCell;
            point_Info newPoint;
            
            for (uint16_t c = 0; c < pointCount; c++){
                point_index++;
                vec3f point;
                point [0] =  ((float)rand() / RAND_MAX + float(i)) / float(subdivision);
                point [1] =  ((float)rand() / RAND_MAX + float(j)) / float(subdivision);
                point [2] = float(gridLayer);
            
                currentCell.points.push_back(point);
           
                float weight;
                // Randomized weight for testing
                if (gridLayer == 0){
                    newPoint.points_weight  = (((float)rand() / (RAND_MAX)) + 0.01) * 1.0 / (1.4 * INIT_SUBDIV);
                    tree_index++;
                    newPoint.tree_index = tree_index;
                }
                else{
                    weight = 1.0f;
                    newPoint.tree_index = -1;

                }

                newPoint.global_point_index = point_index;    

                currentCell.pointsInfo.push_back(newPoint);
                // currentCell.pointsInfo.push_back(weight);
            }
            currentCellRow.push_back(currentCell);
            currentCellPointsCount.push_back(pointCount);
        }
        grid.cells.push_back(currentCellRow);
        grid.pointsCount.push_back(currentCellPointsCount);
    }

    return grid;
    
}


Coord getClosestPoint(const Grid2D& grid, const vec3f& point, const  uint32_t gridLayer){

    // get the corresponding cell in the lower level projected from the current point
    vec2i cell({int(point[0] * grid.cells[0].size() ),int(point[1] * grid.cells.size())});
    Coord closestPoint;
    float mindistsqrd = 10.0f;

    for(int j = cell[1]-1; j <= cell[1] +1 ; j++){
        for(int i = cell[0]-1; i <=cell[0]+1 ; i++){
            // make sure the cells that are being checked is within the boundary of the grids
            if( i >= 0 &&  i < grid.cells[0].size() &&  j >= 0 && j < grid.cells.size()){

                const Cell *currentCell = &(grid.cells[j][i]);
                for (int p = 0; p < grid.cells[j][i].points.size(); p++){
                    
                    // d² 
                    float distsqrd  = dot(point - currentCell->points[p],  point - currentCell->points[p]);

                    float power = distsqrd - pow(currentCell->pointsInfo[p].points_weight, 2.0);

                    if(power < mindistsqrd){
                        mindistsqrd = power;
 
                        closestPoint.gridIndex = gridLayer;
                        closestPoint.coord[0] = i;
                        closestPoint.coord[1] = j;
                        closestPoint.pointIndex = p;
                        closestPoint.weight = currentCell->pointsInfo[p].points_weight;
                        closestPoint.tree_index = currentCell->pointsInfo[p].tree_index;
                    }
                }
            }
        }
    }

    return closestPoint;

}