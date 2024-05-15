#include "ultils.h"


void filter_trees(std::vector<Tree> &trees){
    for (int i = 0; i < trees.size(); i++){
        if (trees[i].numEdges< 100){
            trees[i].numEdges = -1;
        }
    }
}


int gridZeroPointsCount = 0;

uint32_t coordToIndex(const Coord &c, const std::vector<Grid2D> &grids){
    int gridIndex = c.gridIndex;
    int x = c.coord[0];
    int y = c.coord[1];
    int p = c.pointIndex;

    return grids[gridIndex].cells[y][x].pointsInfo[p].global_point_index + gridZeroPointsCount;
}

void branch_styling(std::vector<Grid2D> &grids, std::vector<Edge> &edges, std::vector<vec3f> &points, std::vector<int> &trees){

    for(int  i = 0 ; i < edges.size(); i++ ){
        auto e = edges[i];

        if (trees[e.c1.tree_index] != -1){
     
            int i1 = coordToIndex(e.c1, grids);
            int i2 = coordToIndex(e.c2, grids);
    
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


void write_to_OBJ(std::vector<Grid2D> grids, std::vector<Edge> edges, std::vector<vec3f> points, std::vector<Tree> &trees){
    // Write to OBJ
    std::cout<<(edges.size())<<"\n";
    std::ofstream ofs;
    ofs.open("graph.obj", std::ofstream::out | std::ofstream::trunc);

    // for (int k = 0; k < points.size(); k++) {
    //     ofs << "v " << points[k][0] << " " << points[k][1] << " " <<  points[k][2]  << "\n";
    // }

    // for (auto e: edges) {
    //     if (trees[e.c1.tree_index].numEdges != -1){
    //         int i1 = coordToIndex(e.c1, grids);
    //         int i2 = coordToIndex(e.c2, grids);
    
    //         ofs << "l " << i1+1 << " " << i2+1 << "\n";
    //     }
        
    // }
    
    // // connecting the root layer to the zeroth layer

    // for (int k = 0; k < gridZeroPointsCount; k++) {

    //     ofs << "l " << k+1 << " " << k+1 + gridZeroPointsCount << "\n";
    // }

    for (int i = 0; i < trees.size(); i++){
        Tree *current_tree = &trees[i];

        if (current_tree->numEdges != -1){
            ofs << "g " << "Tree_"<< std::to_string(trees[i].ID ) << std::endl;

            for (int e = 0; e < current_tree->numEdges; e++){
                Edge *current_edge = current_tree->branches[e];

                int i1 = coordToIndex(current_edge->c1, grids);
                int i2 = coordToIndex(current_edge->c2, grids);

                ofs << "l " << i1+1 << " " << i2+1 << "\n";
            }

        }
    }
    ofs.close();
}