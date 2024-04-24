#include "jitteredGrid.h"
#include <random>

using namespace LavaCake;

// DENSITY --------------------------------------------------------------
std::vector<std::vector<float>> random_density_map (int dense_region_count, int subdiv){

    std::vector<LavaCake::vec2i> dense_centers;
    std::vector<std::vector<float>> map;

    // Pick randomly dense_region_count cells in the current grid as density center
    if (dense_region_count > subdiv* subdiv){
        dense_region_count = subdiv * subdiv;
    }

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
            float coeff = 1.0 - 4.0 * sqrt(minDistSqr)/(subdiv * sqrt(2));

            if (coeff < 0) coeff = 0;
            currentRow.push_back(coeff);
        }
        map.push_back(currentRow);
    }
    
    return map;
    
}

std::vector<std::vector<float>> user_density_map(std::string filename, int subdiv){
    int width, height, channelsNum;
    int desiredChannels = 1; // grayscale

    unsigned char * image = stbi_load(filename.c_str(), &width, &height, &channelsNum, desiredChannels);

    if (image == NULL){
        std::cout << "Failed to load density map\n" << std::endl;
        exit(1);
    }

    unsigned char * resized_im = stbir_resize_uint8_srgb(image, width, height, 0, NULL, subdiv, subdiv, 0, STBIR_1CHANNEL);

    std::vector<std::vector<float>> map;

    for (int j = 0; j < subdiv; j++){
        std::vector<float> currentRow;
        for (int i = 0; i < subdiv; i++){
            currentRow.push_back(1.0 - float(int(resized_im[j * subdiv + i])/255.0));
        }
        map.push_back(currentRow);
        
    }

    return map;
}

// ----------------------------------------------------------------------------



/*
    GenerateGrid:   randomize points line by line through an
                    abstractly subdivided grid
                    Currently: 1 point per cell per grid
*/
u_int16_t maxPointsPerCell = 8;
int tree_index = -1;

Grid2D generateGrid(u_int16_t subdivision, int seed, int gridLayer, std::string filename, int &point_index){

    Grid2D grid;
    srand(seed + 124534);

    std::vector<std::vector<float>> weight_map;

    if (!filename.empty()){
        weight_map = user_density_map(filename, subdivision);
    }
    
    float init_subdiv = 2;
    for(u_int16_t  j = 0; j < subdivision ; j++ ){
        std::vector<Cell> currentCellRow;
        std::vector<int> currentCellPointsCount;

        for(u_int16_t  i = 0; i < subdivision ; i++ ){

            int pointCount;

            if (!filename.empty()){
                pointCount = int(float(MAX_POINT_PER_CELL) * weight_map[j][i]);
            }
            else{
                pointCount = MAX_POINT_PER_CELL;
            }


            Cell currentCell;
            point_Info newPoint;
            
            for (u_int16_t c = 0; c < pointCount; c++){
                point_index++;
                vec3f point;
                point [0] =  ((float)rand() / RAND_MAX + float(i)) / float(subdivision);
                point [1] =  ((float)rand() / RAND_MAX + float(j)) / float(subdivision);
                point [2] = float(gridLayer);
            
                currentCell.points.push_back(point);
           
                float weight;
                // Randomized weight for testing
                if (gridLayer == 0){
                    newPoint.points_weight  = (float)rand() / (RAND_MAX+1.0) *0.15 + 0.1;
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

    for(int j = cell[1]-5; j <= cell[1] +5 ; j++){
        for(int i = cell[0]-5; i <=cell[0]+5 ; i++){
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
                        closestPoint.global_index = currentCell->pointsInfo[p].global_point_index;
                        closestPoint.weight = currentCell->pointsInfo[p].points_weight;
                        closestPoint.tree_index = currentCell->pointsInfo[p].tree_index;
                    }
                }
            }
        }
    }

    // std::cout << closestPoint.weight << " " << closestPoint.tree_index << std::endl;

    return closestPoint;

}