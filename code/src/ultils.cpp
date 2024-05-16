#include "ultils.h"


void filter_trees(std::vector<Tree> &trees){
    for (int i = 0; i < trees.size(); i++){
        if (trees[i].numBranches< 100){
            trees[i].numBranches = -1;
        }
    }
}


int gridZeroPointsCount = 0;

Point pointFromCoord(const Coord &c, const std::vector<Grid2D> &grids){
    int gridIndex = c.gridIndex;
    int x = c.coord[0];
    int y = c.coord[1];
    int p = c.pointIndex;
    
    Point newPoint;

    vec3f position = grids[gridIndex].cells[y][x].points[p];

    int returnIndex = grids[gridIndex].cells[y][x].pointsInfo[p].global_point_index + gridZeroPointsCount;

    newPoint.position = position;
    newPoint.index = returnIndex;

    return newPoint;
}

uint32_t coordToIndex(const Coord &c, const std::vector<Grid2D> &grids){
    int gridIndex = c.gridIndex;
    int x = c.coord[0];
    int y = c.coord[1];
    int p = c.pointIndex;

    int returnIndex = grids[gridIndex].cells[y][x].pointsInfo[p].global_point_index + gridZeroPointsCount;

    return returnIndex;
}

void branch_styling(std::vector<Grid2D> &grids, std::vector<vec3f> &points, std::vector<Tree> &trees){

    for (int i = 0; i < trees.size(); i++){
        Tree *current_tree = &trees[i];
        if (current_tree->numBranches != -1){
            for (int e = 0; e < current_tree->numBranches; e++){
                Edge *current_edge = &current_tree->edges[e];

                int i1 = coordToIndex(current_edge->c1, grids);
                int i2 = coordToIndex(current_edge->c2, grids);
                
                vec3f point1 = points[i1];
                vec3f &point2 = points[i2];

                auto delta = point2-point1;
                float l2 = delta[0]*delta[0] + delta[1]*delta[1];

                float edgeLength = 1.0f/pow(FLATNESS,current_edge->c2.gridIndex) ;

                float deltasqrd = edgeLength*edgeLength - l2 ;

                deltasqrd = deltasqrd < 0.0f ?  0.01f : deltasqrd;
                point2[2] = point1[2] + sqrt(deltasqrd);
            }
        }
    }
}


void write_to_OBJ(std::vector<Grid2D> grids, std::vector<vec3f> points, std::vector<Tree> &trees){
    // Write to OBJ
    std::ofstream ofs;
    ofs.open("graph.obj", std::ofstream::out | std::ofstream::trunc);

    for (int k = 0; k < points.size(); k++) {
        ofs << "v " << points[k][0] << " " << points[k][1] << " " <<  points[k][2]  << "\n";
    }

    int count = -1;
    for (int i = 0; i < trees.size(); i++){
        Tree *current_tree = &trees[i];

        if (current_tree->numBranches != -1){
            count++;
            
            ofs << "o " << "Tree_"<< std::to_string(trees[i].ID) << "\n";

            for (int e = 0; e < current_tree->numBranches; e++){
                Edge *current_edge = &current_tree->edges[e];

                int i1 = coordToIndex(current_edge->c1, grids);
                int i2 = coordToIndex(current_edge->c2, grids);

                ofs << "l " << i1+1 << " " << i2+1 << "\n";
            }

            ofs << "l " << count+1 << " " << count+1+ gridZeroPointsCount << "\n";

            ofs << "\n";
        }

    }
    ofs.close();
}