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

    // Generate grids and their corresponding points
    for(int i = 0; i < BRANCHING; i++){
        grids.push_back(generateGrid(int(init_subdiv),(i * 15634) % 3445, i , DENSITY_IMAGE, point_index));

        init_subdiv *= FLATNESS;
    }
    
    // edges: representing the branches of the trees
    std::vector<Edge> edges;

    // Initial trees indicated by root grid layer
    for (int i = 0; i < grids[0].cells.size(); i++){
        gridZeroPointsCount += std::accumulate(grids[0].pointsCount[i].begin(), grids[0].pointsCount[i].end(), 0);
    }
    
    std::vector<int> trees(gridZeroPointsCount);

    /*
    Start from the bottom layer, for each point of the next layer, find and connect to the closest point of the current layer
    */
    for (int k = 0; k < BRANCHING-1; k++){

        for(u_int16_t  j = 0; j < grids[k+1].cells.size() ; j++ ){
            for(u_int16_t  i = 0; i <  grids[k+1].cells[j].size() ; i++ ){   

                Cell *currentCell = &grids[k+1].cells[j][i];
                for (u_int16_t p = 0; p < currentCell->points.size(); p++){

                    // closest point in the lower level
                    Coord c2 = getClosestPoint(grids[k], currentCell->points[p], k);

                    // current point
                    Coord c1;
                    c1.coord = vec2u({u_int32_t(i),u_int32_t(j)});
                    c1.gridIndex = k+1;
                    c1.pointIndex = p;
                    // c2.weight = c1.weight;
                    c1.global_index = currentCell->pointsInfo[p].global_point_index;

                    currentCell->pointsInfo[p].points_weight = c2.weight*WEIGHT_ATTENUATION;
                    currentCell->pointsInfo[p].tree_index = c2.tree_index;

                    edges.push_back({c2,c1});
                    trees[c2.tree_index]++;
                }
            }
        }
    }

// Filtering useless "trees"
std::vector<int> filtered_trees;

// for (int i = 0; i < trees.size(); i++){
//     if (trees[i] < 100){
//         // filtered_trees.push_back(i);
//         trees[i] = -1;
//     }
// }

// Flattening the data structure--------------------------------------------------------
    std::vector<vec3f> points;
    gridZeroPointsCount = 0;    //reset after the trees are filtered
    // Generating 3D points for the root layer
    for(u_int16_t  j = 0; j < grids[0].cells.size(); j++ ){
        for(u_int16_t  i = 0; i <  grids[0].cells[j].size(); i++ ){
            for (u_int16_t p = 0; p < grids[0].cells[j][i].points.size(); p ++){
                int curr_tree_index = grids[0].cells[j][i].pointsInfo[p].tree_index;
                if ( trees[curr_tree_index] != -1){
                    gridZeroPointsCount++;
                    points.push_back(vec3f({grids[0].cells[j][i].points[p][0] *gen_area, grids[0].cells[j][i].points[p][1] * gen_area, float(0)}));
                }
                else{
                    point_index_reduction--;
                }
            }
        }
    }

    // Generate the 3D points for each layer at height of "float(h)" above the initial points in root layer
    
    for (uint16_t k = 0; k < BRANCHING; k++){
        for(u_int16_t  j = 0; j< grids[k].cells.size(); j++ ){   
            for(u_int16_t  i = 0; i <  grids[k].cells[j].size() ; i++ ){
                for (u_int16_t p = 0; p < grids[k].cells[j][i].points.size(); p ++){
                    int curr_tree_index = grids[k].cells[j][i].pointsInfo[p].tree_index;
                    if ( trees[curr_tree_index] != -1){
                        points.push_back(vec3f({grids[k].cells[j][i].points[p][0] * gen_area, grids[k].cells[j][i].points[p][1] * gen_area, float(k+1)}));  
                    }
                    else{
                        point_index_reduction--;
                    }
                }
            }
        }
    }

for (auto i:trees) std::cout << i << " ";
std::cout << std::endl;

for (auto i:filtered_trees) std::cout << i << " ";
std::cout << std::endl;
std::cout << gridZeroPointsCount << " "<< point_index << " " << point_index_reduction << std::endl;

//----------------------------------------------------------------------------------------

    // branch_styling(&grids, &edges, &points, point_index_reduction, gridZeroPointsCount);
    // write_to_OBJ(grids, edges, points, point_index_reduction, gridZeroPointsCount);

    branch_styling(&grids, &edges, &points);
    write_to_OBJ(grids, edges, points);

    return 0;
}