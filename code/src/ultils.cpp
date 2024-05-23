#include "ultils.h"


void filter_trees(std::vector<Tree> &trees){
    for (int i = 0; i < trees.size(); i++){
        if (trees[i].numBranches < BRANCHES_COUNT_THRESHOLD){
            trees[i].numBranches = -1;
        }
    }
}


int gridZeroPointsCount = 0;
std::pair<int, Point> pointFromCoord(const Coord &c, const std::vector<Grid2D> &grids){
    int gridIndex = c.gridIndex;
    int x = c.coord[0];
    int y = c.coord[1];
    int p = c.pointIndex;
    
    Point newPoint;

    vec3f position = grids[gridIndex].cells[y][x].points[p];
    position[0] *= GEN_AREA;
    position[1] *= GEN_AREA;

    int returnIndex = grids[gridIndex].cells[y][x].pointsInfo[p].global_point_index + gridZeroPointsCount;

    newPoint.position = position;
  

    return std::make_pair(returnIndex, newPoint);
}


// DENSITY --------------------------------------------------------------

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

// CROWNSHYNESS EFFECT ------------------------------------------------
void crownShyness(std::vector<vec3f> &points, std::vector<Tree>&trees){
    std::vector<std::vector<float>> shrink_map;
    std::string shrink_factor_image = SHRINK_FACTOR_IMAGE;
    if (!shrink_factor_image.empty()){
        shrink_map = user_density_map(shrink_factor_image, INIT_SUBDIV);
    }
    
    for (auto t: trees){
        if (t.numBranches != -1){
            int x = points[(*t.points.begin()).first][0];
            int y = points[(*t.points.begin()).first][1];

            t.center[0] /= t.points.size();
            t.center[1] /= t.points.size();
            t.center[2] /= t.points.size();

            float shrink_factor = DEFAULT_TREE_SHRINK_FACTOR;
            if (!shrink_factor_image.empty()){
                shrink_factor = shrink_map[y][x];
            }

            for (auto it = t.points.begin(); it != t.points.end(); it++){
                points[(*it).first][0] = (points[(*it).first][0] - t.center[0]) * shrink_factor + t.center[0];
                points[(*it).first][1] = (points[(*it).first][1] - t.center[1]) * shrink_factor + t.center[1];
            }
        }
    }
}


void branch_styling(std::vector<vec3f> &points, std::vector<Tree> &trees){

    for (int i = 0; i < trees.size(); i++){
        Tree *current_tree = &trees[i];
        if (current_tree->numBranches != -1){
            for (int e = 0; e < current_tree->numBranches; e++){
                Branch *current_branch = &current_tree->branches[e];

                vec3f point1 = points[current_branch->i1];
                vec3f &point2 = points[current_branch->i2];

                auto delta = point2-point1;
                float l2 = delta[0]*delta[0] + delta[1]*delta[1];

                float edgeLength = 1.0f/pow(FLATNESS, point2[2]-1) ;

                float deltasqrd = edgeLength*edgeLength - l2 ;

                deltasqrd = deltasqrd < 0.0f ?  0.01f : deltasqrd;
                point2[2] = point1[2] + sqrt(deltasqrd) * (0.5 * (float)rand() /RAND_MAX + 1.0);
            }
        }
    }
}


void write_to_OBJ(std::vector<vec3f> points, std::vector<Tree> &trees){
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

            // Writing the edges
            for (int e = 0; e < current_tree->numBranches; e++){
                
                Branch *current_branch = &current_tree->branches[e];

                ofs << "l " << (current_branch->i1)+1 << " " << (current_branch->i2)+1 << "\n"; 
            }

            ofs << "l " << count+1 << " " << count+1+ gridZeroPointsCount << "\n";

            ofs << "\n";
        }

    }
    ofs.close();
}