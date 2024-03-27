#include "jitteredGrid.h"
#include <iostream>
#include <fstream> 
#include <math.h>      

#include "ultils.h"
#define BRANCHING 5

using namespace LavaCake;


int main(){
    // grids: represent the layers of branch deviations (by height)
    std::vector<Grid2D> grids;
    
    // Indicate the subdivision size of the current space
    // i.e tile the space into 10x10 grid
    float subdiv = 5;

    float init_subdiv = 2; // area of generation (e.g 20 means 20x20 m^2)

    float flatness = 2;

    int dense_region_count = 1; // Default number of dense clusters in the generation area

    // Generate grids and their corresponding points
    std::string filename = "/home/local/canopy_forest/crownshyness/code/testing";
    for(int i = 0; i < BRANCHING; i++){

        // grids.push_back(generateGrid(int(subdiv),(i * 15634) % 3445, i , filename+ std::to_string(i) + ".png"));

        grids.push_back(generateGrid(int(subdiv),(i * 15634) % 3445, i, ""));

        // increase the subdivision at the next layer
        subdiv = subdiv * flatness;
    }
    
    // edges: representing the branches of the trees
    std::vector<Edge> edges;

    /*
    Start from the bottom layer, for each point of the nextlayer, find and connect to the closest point of the current layer
    */

    for (int k = 0; k < BRANCHING-1; k++){

        for(u_int16_t  j = 0; j < grids[k+1].cells.size() ; j++ ){
            for(u_int16_t  i = 0; i <  grids[k+1].cells[j].size() ; i++ ){   

                Cell currentCell = grids[k+1].cells[j][i];
                for (u_int16_t p = 0; p < currentCell.points.size(); p++){

                    // closest point in the lower level
                    Coord c2 = getClosestPoint(grids[k], currentCell.points[p], k);

                    // current point
                    Coord c1;
                    c1.coord = vec2u({u_int32_t(i),u_int32_t(j)});
                    c1.gridIndex = k+1;
                    c1.pointIndex = p;
                    // c2.weight = c1.weight;

                    currentCell.points_weight[p] = c2.weight;

                    edges.push_back({c2,c1});
                }
            }
        }
    }

// Flattening the data structure--------------------------------------------------------
    std::vector<vec3f> points;

    // Generating 3D points for the root layer
    for(u_int16_t  j = 0; j < grids[0].cells.size(); j++ ){
        for(u_int16_t  i = 0; i <  grids[0].cells[j].size(); i++ ){
            for (u_int16_t p = 0; p < grids[0].cells[j][i].points.size(); p ++){
                points.push_back(vec3f({grids[0].cells[j][i].points[p][0] *init_subdiv, grids[0].cells[j][i].points[p][1] * init_subdiv, float(0)}));
            }
        }
    }

    // Generate the 3D points for each layer at height of "float(h)" above the initial points in root layer
    
    for (uint16_t k = 0; k < BRANCHING; k++){
        for(u_int16_t  j = 0; j< grids[k].cells.size(); j++ ){   
            for(u_int16_t  i = 0; i <  grids[k].cells[j].size() ; i++ ){
                for (u_int16_t p = 0; p < grids[k].cells[j][i].points.size(); p ++){

                    points.push_back(vec3f({grids[k].cells[j][i].points[p][0] *init_subdiv, grids[k].cells[j][i].points[p][1] * init_subdiv, float(k+1)}));  
                }
            }
        }
    }

//----------------------------------------------------------------------------------------

    branch_styling(&grids, &edges, &points);
    write_to_OBJ(grids, edges, points);

    return 0;
}