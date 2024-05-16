#include "jitteredGrid.h"
#include <iostream>
#include <fstream> 
#include <math.h>      

#include "ultils.h"

using namespace LavaCake;


int main(){
    // grids: represent the layers of branch deviations (by height)
    std::vector<Grid2D> grids;
    
    float init_subdiv = INIT_SUBDIV;
    float gen_area = GEN_AREA; 
    int point_index = -1;
    int dense_region_count = 1; // Default number of dense clusters in the generation area
    
    std::cout << "Generating forest..." << std::endl;

    // Generate grids and their corresponding points
    for(int i = 0; i < BRANCHING; i++){
        grids.push_back(generateGrid(int(init_subdiv),(i * 15634) % 3445, i , DENSITY_IMAGE, point_index));

        init_subdiv *= FLATNESS;
    }
    

    // Initial trees indicated by root grid layer
    for (int i = 0; i < grids[0].cells.size(); i++){
        gridZeroPointsCount += std::accumulate(grids[0].pointsCount[i].begin(), grids[0].pointsCount[i].end(), 0);
    }
    
    // Allocate a list for the generated trees
    std::vector<Tree> trees(gridZeroPointsCount);

    for (int i = 0; i < gridZeroPointsCount; i++){
        trees[i].ID = i;
    }

    /*
    Start from the bottom layer, for each point of the next layer, find and connect to the closest point of the current layer
    */
    for (int k = 0; k < BRANCHING-1; k++){

        for(uint16_t  j = 0; j < grids[k+1].cells.size() ; j++ ){
            for(uint16_t  i = 0; i <  grids[k+1].cells[j].size() ; i++ ){   

                Cell *currentCell = &grids[k+1].cells[j][i];
                for (uint16_t p = 0; p < currentCell->points.size(); p++){

                    // closest point in the lower level
                    Coord c2 = getClosestPoint(grids[k], currentCell->points[p], k);

                    // current point
                    Coord c1;
                    c1.coord = vec2u({uint32_t(i),uint32_t(j)});
                    c1.gridIndex = k+1;
                    c1.pointIndex = p;
                    // c2.weight = c1.weight;
                    currentCell->pointsInfo[p].points_weight = c2.weight*WEIGHT_ATTENUATION;
                    currentCell->pointsInfo[p].tree_index = c2.tree_index;

                    trees[c2.tree_index].numEdges++;
                    trees[c2.tree_index].branches.push_back({c2,c1});
                }
            }
        }
    }

// Filtering useless "trees"

if (FILTER_TREES == true){
    filter_trees(trees);
}

// Flattening the data structure--------------------------------------------------------
    std::vector<vec3f> points;
    gridZeroPointsCount = 0;    //reset after the trees are filtered
    // Generating 3D points for the root layer
    for(uint16_t  j = 0; j < grids[0].cells.size(); j++ ){
        for(uint16_t  i = 0; i <  grids[0].cells[j].size(); i++ ){
            for (uint16_t p = 0; p < grids[0].cells[j][i].points.size(); p ++){
                int curr_tree_index = grids[0].cells[j][i].pointsInfo[p].tree_index;
                if ( trees[curr_tree_index].numEdges != -1){
                    gridZeroPointsCount++;
                    points.push_back(vec3f({grids[0].cells[j][i].points[p][0] *gen_area, grids[0].cells[j][i].points[p][1] * gen_area, float(0)}));
                }
            }
        }
    }

    // Generate the 3D points for each layer at height of "float(h)" above the initial points in root layer
    
    point_index = -1; // reset point index for filtering trees
    for (uint16_t k = 0; k < BRANCHING; k++){
        for(uint16_t  j = 0; j< grids[k].cells.size(); j++ ){   
            for(uint16_t  i = 0; i <  grids[k].cells[j].size() ; i++ ){
                for (uint16_t p = 0; p < grids[k].cells[j][i].points.size(); p ++){
                    int curr_tree_index = grids[k].cells[j][i].pointsInfo[p].tree_index;
                    if ( trees[curr_tree_index].numEdges != -1){
                        point_index++;
                        points.push_back(vec3f({grids[k].cells[j][i].points[p][0] * gen_area, grids[k].cells[j][i].points[p][1] * gen_area, float(k+1)}));  

                        grids[k].cells[j][i].pointsInfo[p].global_point_index = point_index; // update point index after filtering
                    }
                }
            }
        }
    }


//----------------------------------------------------------------------------------------
    if (BRANCH_STYLING == true){
        branch_styling(grids, points, trees);
    }

    std::cout << "Writing to OBJ..." << std::endl;
    write_to_OBJ(grids, points, trees);

    std::cout << "All done !!" << std::endl;

    return 0;
}