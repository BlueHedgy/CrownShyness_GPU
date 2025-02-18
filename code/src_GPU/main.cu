#include <iostream>
#include <fstream> 
#include <math.h>      

#include "jitteredGrid_GPU.h"

using namespace LavaCake;

int main(){

    load_Config_Profile(XSTRING(CMAKE_SOURCE_DIR)"/config/config.json"); 

    // Represent the layers of branch deviations
    std::vector<Grid2D> grids;

    
    float subdiv = INIT_SUBDIV;
    float gen_area = GEN_AREA; 
    int point_index = -1;

    std::cout << "Generating forest..." << std::endl;
    
    generateGrid_GPU(subdiv, 1, 1, DENSITY_IMAGE, point_index);
    
    // Generate grids and their corresponding points
    // for(int i = 0; i < BRANCHING; i++){
    //     grids.push_back(generateGrid_GPU(int(subdiv),(i * 15634) % 3445, i, DENSITY_IMAGE, point_index));

    //     subdiv *= SCALE;
    // }

//     // Temporary form for the branches of the trees    
//     std::vector<Edge> edges;

//     // Initial number of trees indicated by root grid layer
//     for (int i = 0; i < grids[0].cells.size(); i++){
//         gridZeroPointsCount += std::accumulate(grids[0].pointsCount[i].begin(), grids[0].pointsCount[i].end(), 0);
//     }
    
//     // Allocate a list for the generated trees
//     std::vector<Tree> trees(gridZeroPointsCount);

//     for (int i = 0; i < gridZeroPointsCount; i++){
//         trees[i].ID = i;
//     }

//     /*
//     For each point of the directly higher layer, find the closest point in the current layer and mark them as an Edge
//     */
//     for (int k = 0; k < BRANCHING-1; k++){

//         for(uint16_t  j = 0; j < grids[k+1].cells.size(); j++){
//             for(uint16_t  i = 0; i <  grids[k+1].cells[j].size(); i++){   

//                 Cell *currentCell = &grids[k+1].cells[j][i];
//                 for (uint16_t p = 0; p < currentCell->points.size(); p++){

//                     // closest point in the lower level
//                     Coord c2 = getClosestPoint(grids[k], currentCell->points[p], k);

//                     // current point
//                     Coord c1;
//                     c1.coord = vec2u({uint32_t(i),uint32_t(j)});
//                     c1.gridIndex = k+1;
//                     c1.pointIndex = p;
//                     c1.tree_index = c2.tree_index;

//                     currentCell->pointsInfo[p].points_weight = c2.weight*WEIGHT_ATTENUATION;
//                     currentCell->pointsInfo[p].tree_index = c2.tree_index;
//                     currentCell->pointsInfo[p].strength = c2.strength * 0.3f;

//                     trees[c2.tree_index].numBranches++;
//                     edges.push_back({c2,c1});
                    
//                 }
//             }
//         }
//     }

//     // Filtering useless "trees"

//     if (FILTER_TREES == true){
//         filter_trees(trees);
//     }


// // Flattening the data structure--------------------------------------------------------
//     std::vector<vec3f> points;
//     gridZeroPointsCount = 0;    //reset after the trees are filtered
    
//     // Generating 3D points for the root layer
//     for(uint16_t j = 0; j < grids[0].cells.size(); j++){
//         for(uint16_t i = 0; i < grids[0].cells[j].size(); i++){

//             Cell *current_cell = &grids[0].cells[j][i];
//             for (uint16_t p = 0; p < current_cell->points.size(); p++){

//                 int curr_tree_index = current_cell->pointsInfo[p].tree_index;
//                 if ( trees[curr_tree_index].numBranches != -1){
//                     gridZeroPointsCount++;
//                     // shift also a bit the root point to create leaning effect of the tree trunks
// 					points.push_back(
//                         vec3f({
//                         (current_cell->points[p][0] + (1.0f/ (MAX_POINT_PER_CELL * INIT_SUBDIV)) * (2.0f * (float) rand() / RAND_MAX - 1.0f)) * gen_area,
// 						(current_cell->points[p][1] + (1.0f/ (MAX_POINT_PER_CELL * INIT_SUBDIV)) * (2.0f * (float) rand() / RAND_MAX - 1.0f)) * gen_area,
//                         float(0)})
//                         );
// 				}
//             }
//         }
//     }

//     // Generate the 3D points for each layer at height of "float(h)" above the initial points in root layer
    
//     point_index = -1; // reset point index for filtering trees
//     for (uint16_t k = 0; k < BRANCHING; k++){
//         for(uint16_t j = 0; j< grids[k].cells.size(); j++){   
//             for(uint16_t  i = 0; i < grids[k].cells[j].size(); i++){

//                 Cell *current_cell = &grids[k].cells[j][i];
//                 for (uint16_t p = 0; p < current_cell->points.size(); p++){

//                     int curr_tree_index = current_cell->pointsInfo[p].tree_index;
//                     if ( trees[curr_tree_index].numBranches != -1){
//                         point_index++;
//                         if (k != 0){
//                             points.push_back(
//                                 vec3f({
//                                 current_cell->points[p][0] * gen_area,
//                                 current_cell->points[p][1] * gen_area,
//                                 float(k+1)})
//                                 );
//                         }
//                         else{
//                             points.push_back(
//                                 vec3f({
//                                 current_cell->points[p][0] * gen_area,
//                                 current_cell->points[p][1] * gen_area,
//                                 float(0.3 * (float)rand() / RAND_MAX + 0.7)})
//                                 );    
//                         }

// 						current_cell->pointsInfo[p].global_point_index = point_index; // update point index after filtering
//                     }
//                 }
//             }
//         }
//     }


// // Process the edges into branches for the trees
//     for (int i = 0; i < edges.size(); i++){
//         Edge &e = edges[i];
//         if (trees[e.c2.tree_index].numBranches != -1){
//             std::pair<int, Point> iP1 = pointFromCoord(e.c1, grids, points);
//             std::pair<int, Point> iP2 = pointFromCoord(e.c2, grids, points);

//             if (trees[e.c2.tree_index].points.insert(iP1).second == true){
//                 trees[e.c2.tree_index].center = trees[e.c2.tree_index].center + points[iP1.first];
//             }
            
//             // accumulating the total 3d position of the points in the tree
//             if (trees[e.c2.tree_index].points.insert(iP2).second == true){
//                 trees[e.c2.tree_index].center = trees[e.c2.tree_index].center + points[iP2.first];
//             }

//             trees[e.c2.tree_index].points.at(iP1.first).children.push_back(iP2.first);
            
//             trees[e.c2.tree_index].points.at(iP2.first).parent = iP1.first;
//             trees[e.c2.tree_index].points.at(iP2.first).direction = normalize(points[iP2.first] - points[iP1.first]);

//             trees[e.c2.tree_index].branches.push_back(Branch(iP2.first, iP1.first, iP2.second.grid_index, iP1.second.grid_index));
//         }
//     }

// //----------------------------------------------------------------------------------------

// /* LEGACY BRANCH STYLING, NOW CURRENTLY USED IN edgeToSpline */
//     // if (BRANCH_STYLING == true){
//     //     branch_styling(points, trees);
//     // }

// /* UNCOMMENT THE ABOVE TO GO BACK TO STRAIGHT EDGES--------- */

//     forest_height(points, trees);

//     // Scaling the trees for the crownshyness effect
//     crownShyness(points, trees);
    
//     edgeToSpline(points, trees);    // comment this out also for straight edges version

//     std::cout << points.size() << " VERTICES" << std::endl;

//     std::cout << "Writing to OBJ..." << std::endl;
//     write_to_OBJ(points, trees);

    std::cout << "All done !!" << std::endl;

    return 0;
}