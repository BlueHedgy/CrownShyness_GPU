#include "ultils.h"

uint32_t coordToIndex(const Coord & c, const std::vector<Grid2D>& grids){
    uint32_t index = 0;
    
    for (int i = 0; i < grids[0].cells.size(); i++){
        for (int j = 0; j < grids[0].cells[i].size(); j++){
            for (int p = 0; p < grids[0].cells[i][j].points.size(); p++){
                index ++;
            }
        }
    }


    for(int k = 0; k < int(c.gridIndex); k++){
        for (int i = 0; i < grids[k].cells.size(); i++){
            for (int j = 0; j < grids[k].cells[i].size(); j++){
                for (int p = 0; p < grids[k].cells[i][j].points.size(); p++){
                    index ++;
                }
            }
        }
    }
    int gridStart = index;

    for (int i = 0; i < int(c.coord[1]); i++){
        for (int j = 0; j < grids[c.gridIndex].cells[i].size(); j++){
            for (int p = 0; p < grids[c.gridIndex].cells[i][j].points.size(); p++){

                index ++;
            }
        }
    }

    int rowStart = index;

    for (int j = 0; j < int(c.coord[0]); j++){
        for (int p = 0; p < grids[c.gridIndex].cells[c.coord[1]][j].points.size(); p++){
            index ++;
        }
    }
    int cellStart = index;
    index+= c.pointIndex;
    
    return index;
}


void write_to_OBJ(std::vector<Grid2D> grids, std::vector<Edge> edges, std::vector<vec3f> points){
    // Write to OBJ
    std::cout<<(edges.size())<<"\n";
    std::ofstream ofs;
    ofs.open("graph.obj", std::ofstream::out | std::ofstream::trunc);

    

    for (int k = 0; k < points.size(); k++) {
        ofs << "v " << points[k][0] << " " << points[k][1] << " " <<  points[k][2]  << "\n";
    }

    for (auto e: edges) {
        auto i1 = coordToIndex(e.c1,grids) + 1;
        auto i2 = coordToIndex(e.c2,grids) + 1;
        ofs << "l " << i1 << " " << i2  << "\n";
    }
    
    // connecting the root layer to the zeroth layer
    int gridZeroPointscount = 0;
    for (int i = 0; i < grids[0].cells.size(); i++){
        gridZeroPointscount += std::accumulate(grids[0].pointsCount[i].begin(), grids[0].pointsCount[i].end(), 0);
    }

    for (int k = 0; k < gridZeroPointscount; k++) {

        ofs << "l " << k+1 << " " << k+1 + gridZeroPointscount << "\n";
    }

    ofs.close();
}