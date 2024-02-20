#include "jitteredGrid.h"
#include <iostream>
#include <fstream> 
#include <math.h>      


#define BRANCHING 5

using namespace LavaCake;


uint32_t coordToIndex(const Coord & c, const std::vector<Grid2D>& grids){
    uint index = grids[0].cells.size() *  grids[0].cells[0].size();
    for(int i = 0; i <c.gridIndex; i++){
        index += grids[i].cells.size() *  grids[i].cells[0].size();
    }

    return index + c.coord[1] * grids[c.gridIndex].cells.size() +  c.coord[0];
}

int main(){
    // grids: represent the layers of branch deviations (by height)
    std::vector<Grid2D> grids;
    
    // Indicate the subdivision size of the current space
    // i.e tile the space into 10x10 grid

    float subdiv = 10;

    float init_subdiv = 10; // area of generation (e.g 20 means 20x20 m^2)
    
    // int gridSubdiv = 4;

    float flatness = 2;

    int dense_region_count = 1; // Default number of dense clusters in the generation area

    // Generate grids and their corresponding points
    for(int i = 0; i < BRANCHING; i++){
        grids.push_back(generateGrid(int(subdiv),(i * 1562434) % 3445 ));

        // increase the subdivision at the next layer
        subdiv = subdiv * flatness;
    }
    
    // edges: representing the branches of the trees
    std::vector<Edge> edges;

    /*
    Start from the top layer, for each point of each layer, find and connect to the closest point of the layer below
    */
    for(int k = BRANCHING-1; k >= 1; k--){
        // iterating through the cell Rows of the current grid layer
        for(u_int16_t  i = 0; i < grids[k].cells.size() ; i++ ){
            // iterating through the cells in a row
            for(u_int16_t  j = 0; j <  grids[k].cells[i].size() ; j++ ){   

                Cell currentCell = grids[k].cells[i][j];
                // iterating through the points in the current cells
                // fix this
                for (u_int16_t p = 0; p < currentCell.points.size(); p++){
                    Coord c1 = getClosestPoint(grids[k-1], grids[k].cells[j][i].points[p], k-1);
                    Coord c2;
                    c2.coord = vec2u({u_int32_t(i),u_int32_t(j)});
                    c2.gridIndex = k;
                    c2.pointIndex = p;

                    edges.push_back({c1,c2});
                }
                
            }
        }
    }

// Flattening the data structure
    std::vector<vec3f> points;

    // Generating 3D points for the root layer
    for(u_int16_t  j = 0; j < grids[0].cells.size() ; j++ ){
        for(u_int16_t  i = 0; i <  grids[0].cells[j].size() ; i++ ){
            for (u_int16_t p = 0; p < grids[0].cells[j][i].points.size(); p ++){

                points.push_back(vec3f({grids[0].cells[j][i].points[p][0] *init_subdiv, grids[0].cells[j][i].points[p][1] * init_subdiv, float(0)}));
            }
        }
    }

    // Generate the 3D points for each layer at height of "float(h)" above the initial points in root layer
    
    for (uint16_t k = 0; k < BRANCHING; k++){
        for(u_int16_t  j = 0; j < grids[k].cells.size() ; j++ ){   
            for(u_int16_t  i = 0; i <  grids[k].cells[j].size() ; i++ ){
                for (u_int16_t p = 0; p < grids[0].cells[j][i].points.size(); p ++){
                    points.push_back(vec3f({grids[k].cells[j][i].points[p][0] *init_subdiv, grids[k].cells[j][i].points[p][1] * init_subdiv, float(k)}));  
                }
            }
        }
    }

// ------------------------------------------------------------------------------------------------------------

/*     for(int  i = edges.size() -1; i>= 0; i-- ){
        auto e = edges[i];
        auto i1 = coordToIndex(e.c1,grids) ;
        auto i2 = coordToIndex(e.c2,grids) ;

        vec3f p1 = points[i1];
        vec3f& p2 = points[i2];
        auto delta = p2-p1;
        float l2 = delta[0]*delta[0] + delta[1]*delta[1];

        float edgeLength = 1.0f/pow(flatness,e.c2.gridIndex) ;

        float deltasqrd = edgeLength*edgeLength - l2 ;

        deltasqrd = deltasqrd < 0.0f ?  0.0f : deltasqrd;
        p2[2] = p1[2] + sqrt(deltasqrd);
        
    } */

    // std::cout<<(edges.size())<<"\n";
    // std::ofstream ofs;
    // ofs.open("graph.obj", std::ofstream::out | std::ofstream::trunc);

    

    // for (int k = 0; k < points.size(); k++) {
    //     ofs << "v " << points[k][0] << " " << points[k][1] << " "  <<  points[k][2]  << "\n";
    // }

    // for (auto e: edges) {
    //     auto i1 = coordToIndex(e.c1,grids) + 1;
    //     auto i2 = coordToIndex(e.c2,grids) + 1;
    //     ofs << "l " << i1 << " " << i2  << "\n";
    // }
    
    // for (int k = 0; k < grids[0].cells.size() *  grids[0].cells[0].size(); k++) {
    //     ofs << "l " << k+1 << " " << k+1 + grids[0].cells.size() *  grids[0].cells[0].size() << "\n";
    // }

    // ofs.close();

    return 0;
}