#include "ultils.h"

int gridZeroPointsCount = 0;
int point_index_reduction = 0;

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


void branch_styling(std::vector<Grid2D> &grids, std::vector<Edge> &edges, std::vector<vec3f> &points, std::vector<int> &trees){

    for(int  i = 0 ; i < edges.size(); i++ ){
        auto e = edges[i];
        // auto i1 = coordToIndex(e.c1,*grids) ;
        // auto i2 = coordToIndex(e.c2,*grids) ;

        // int i1 = e.c1.global_index + gridZeroPointsCount;
        // int i2= e.c2.global_index + gridZeroPointsCount;

        if (trees[e.c1.tree_index] != -1){
            int grid1 = e.c1.gridIndex;
            int x1 = e.c1.coord[0];
            int y1 = e.c1.coord[1];
            int p1 = e.c1.pointIndex;

            int grid2 = e.c2.gridIndex;
            int x2 = e.c2.coord[0];
            int y2 = e.c2.coord[1];
            int p2 = e.c2.pointIndex;

            int i1 = grids[grid1].cells[y1][x1].pointsInfo[p1].global_point_index + gridZeroPointsCount;
            int i2 = grids[grid2].cells[y2][x2].pointsInfo[p2].global_point_index + gridZeroPointsCount;
        

            vec3f point1 = points[i1];
            vec3f& point2 = points[i2];

            auto delta = point2-point1;
            float l2 = delta[0]*delta[0] + delta[1]*delta[1];

            float edgeLength = 1.0f/pow(FLATNESS,e.c2.gridIndex) ;

            float deltasqrd = edgeLength*edgeLength - l2 ;

            deltasqrd = deltasqrd < 0.0f ?  0.01f : deltasqrd;
            point2[2] = point1[2] + sqrt(deltasqrd);
        }
        
    }
}


void write_to_OBJ(std::vector<Grid2D> grids, std::vector<Edge> edges, std::vector<vec3f> points, std::vector<int> &trees){
    // Write to OBJ
    std::cout<<(edges.size())<<"\n";
    std::ofstream ofs;
    ofs.open("graph.obj", std::ofstream::out | std::ofstream::trunc);

    for (int k = 0; k < points.size(); k++) {
        ofs << "v " << points[k][0] << " " << points[k][1] << " " <<  points[k][2]  << "\n";
    }

    for (auto e: edges) {
        if (trees[e.c1.tree_index] != -1){
            int grid1 = e.c1.gridIndex;
            int x1 = e.c1.coord[0];
            int y1 = e.c1.coord[1];
            int p1 = e.c1.pointIndex;

            int grid2 = e.c2.gridIndex;
            int x2 = e.c2.coord[0];
            int y2 = e.c2.coord[1];
            int p2 = e.c2.pointIndex;

            int i1 = grids[grid1].cells[y1][x1].pointsInfo[p1].global_point_index + gridZeroPointsCount;
            int i2 = grids[grid2].cells[y2][x2].pointsInfo[p2].global_point_index + gridZeroPointsCount;
            ofs << "l " << i1+1 << " " << i2+1 << "\n";
        }
        
    }
    
    // connecting the root layer to the zeroth layer

    for (int k = 0; k < gridZeroPointsCount; k++) {

        ofs << "l " << k+1 << " " << k+1 + gridZeroPointsCount << "\n";
    }

    ofs.close();
}