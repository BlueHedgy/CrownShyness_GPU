#include "utils.h"

void filter_trees(std::vector<Tree> &trees){
    for (int i = 0; i < trees.size(); i++){
        if (trees[i].numBranches < BRANCHES_COUNT_THRESHOLD){
            trees[i].numBranches = -1;
        }
    }
}


int gridZeroPointsCount = 0;
std::pair<int, Point> pointFromCoord(const Coord &c, const std::vector<Grid2D> &grids, std::vector<vec3f> &points){
    int gridIndex = c.gridIndex;
    int x = c.coord[0];
    int y = c.coord[1];
    int p = c.pointIndex;
    
    Point newPoint;

    int returnIndex = grids[gridIndex].cells[y][x].pointsInfo[p].global_point_index + gridZeroPointsCount;

    newPoint.position = &points[returnIndex];
    newPoint.grid_index = gridIndex;
    newPoint.direction = vec3f({0.0, 0.0, 0.0});
    newPoint.strength = c.strength;


    return std::make_pair(returnIndex, newPoint);
}


// This function purely takes all the points of each tree and move it along z axis based on inputs
void forest_height (std::vector<vec3f> &points, std::vector<Tree> &trees){
    std::vector<std::vector<float>> height_map;
    std::string height_image = FOREST_HEIGHT_IMAGE;

    int width = 1;
    int height = 1;

    float height_coeff = 1.0f;

    if (!height_image.empty()){
        std::cout << "Loaded forest height map" << std::endl;
        height_map = user_density_map(height_image, 0);
        width = height_map[0].size();
        height = height_map.size();
    }

    for (auto t:trees){
        if (t.numBranches != -1){
            float x = points[(*t.points.begin()).first][0];
            float y = points[(*t.points.begin()).first][1];
            float z = points[(*t.points.begin()).first][2];

            if (!height_image.empty()){
                height_coeff = height_map[(int)(y * height / GEN_AREA)][(int)(x * height / GEN_AREA)];
            }

            float delta_z = height_coeff * MAX_FOREST_HEIGHT - z;
            if (delta_z < 0) delta_z = 0;

            for (auto p:t.points){
                float *curr_z = &points[p.first][2];
                (*curr_z) += delta_z;
            }
        }
    }
}


// CROWNSHYNESS EFFECT ------------------------------------------------
void crownShyness(std::vector<vec3f> &points, std::vector<Tree>&trees){
    std::vector<std::vector<float>> shrink_map;
    std::string shrink_factor_image = SHRINK_FACTOR_IMAGE;

    int width = 1; 
    int height = 1;

    // Extracting shrink factor coeff from input texture
    if (!shrink_factor_image.empty()){
        shrink_map = user_density_map(shrink_factor_image, 0);
        width = shrink_map[0].size();
        height = shrink_map.size();
    }

    
    for (auto &t: trees){
        if (t.numBranches != -1){
            float x = points[(*t.points.begin()).first][0];
            float y = points[(*t.points.begin()).first][1];

            // the center was previously accumulated by all the points, so now we get the average
            t.center[0] /= t.points.size();
            t.center[1] /= t.points.size();
            t.center[2] /= t.points.size();

            float shrink_factor = DEFAULT_SHRINK_FACTOR;
            if (!shrink_factor_image.empty()){
                shrink_factor = shrink_map[(int)(y * height / GEN_AREA)][(int)(x * width / GEN_AREA)];
            }
            
            // Shrinking points toward tree center (x,y)
            for (auto it = t.points.begin(); it != t.points.end(); it++){
                points[(*it).first] = (points[(*it).first] - t.center) * shrink_factor + t.center;
            }

            // Further crown shyness in smaller foliage clumps
            for (auto it = t.points.begin(); it != t.points.end(); it++){
                int parent = (*it).second.parent;
                vec3f current_position = *(*it).second.position;

                // Seeking back toward root until the designate layer that this effect 
                // is indicated to occur
                if (parent > -1 && t.points.at(parent).grid_index >= CROWN_SHYNESS_STEP){
                    int shrink_searcher = parent;
                    int shrink_target;

                    while (t.points.at(shrink_searcher).grid_index != CROWN_SHYNESS_STEP){
                        shrink_searcher = t.points.at(shrink_searcher).parent;
                    }

                    shrink_target = shrink_searcher;

                    vec3f shrink_center = *t.points.at(shrink_target).position;

                    // ! Hard coded shrink values here for clumps, try to make it procedural
                    points[(*it).first][0] = (points[(*it).first][0] - shrink_center[0]) * shrink_factor * 0.9 + shrink_center[0];
                    points[(*it).first][1] = (points[(*it).first][1] - shrink_center[1]) * shrink_factor * 0.9 + shrink_center[1];

                }
            }
        }
    }
}


// SIMULATING TREE SILHOUETTES ----------------------------------------------------
void branch_styling(std::vector<vec3f> &points, std::vector<Tree> &trees){
    vec3f z_normal = vec3f({0.0f, 0.0f, 1.0f});
    
    for (int i = 0; i < trees.size(); i++){
        Tree &current_tree = trees[i];

        if (current_tree.numBranches != -1){
            for (int e = 0; e < current_tree.numBranches; e++){

                Branch &current_branch = current_tree.branches[e];
                vec3f &center = current_tree.center;
                
                vec3f point1 = points[current_branch.i1];
                vec3f &point2 = points[current_branch.i2];           

                auto delta = point2-point1;
                float l2 = delta[0]*delta[0] + delta[1]*delta[1];

                float edgeLength = 1.0f/pow(SCALE, current_branch.k2 -1) ;
                
                float deltasqrd = edgeLength*edgeLength - l2 ;

                deltasqrd = deltasqrd < 0.01f ?  0.01f : deltasqrd;
                point2[2] = point1[2] + sqrt(deltasqrd);

            }
        }
    }
}

// Interpolate edges to curves --------------------------------------------

vec3f lerp (vec3f &p1, vec3f &p2, float t){
    const float s = 1.0 - t;
    return vec3f ({ p1[0] * s + p2[0] * t, 
                    p1[1] * s + p2[1] * t, 
                    p1[2] * s + p2[2] * t
                  });
}

vec3f De_Casteljau_Algo(std::vector<vec3f> cPoints, float segment_coeff){
    if (cPoints.size() > 1){
        std::vector<vec3f> new_cPoints;
        for (int p = 0; p < cPoints.size() - 1; p++){
            vec3f new_cPoint = lerp(cPoints[p], cPoints[p+1], segment_coeff);
            new_cPoints.push_back(new_cPoint);
        }

        return De_Casteljau_Algo(new_cPoints, segment_coeff);
    }

    return cPoints[0];

}

void addSplineToTrees(std::vector<vec3f> &points, Tree &t, std::vector<vec3f> &cPoints, int i1, int i2, int numSegments){

    int &splineBranches = t.numSplineBranches;
    int index;

    for (int s = 1; s < numSegments; s++){
        float coeff = ((float)s)/numSegments;
        vec3f pt = De_Casteljau_Algo(cPoints, coeff);

        points.push_back(pt);
        index = points.size() -1;
        if (s == 1) {
            t.spline_Branches.push_back(Branch({i1, index}));
        }
        else if (s == numSegments-1){
            t.spline_Branches.push_back(Branch({index, i2}));
            t.spline_Branches.push_back(Branch({index-1, index}));
            splineBranches++;
        }
        else{
            t.spline_Branches.push_back(Branch({index-1, index}));

        }
        t.points.at(i2).prevIndices.push_back(index);
        splineBranches++;
    }

    t.points.at(i2).prevIndices.push_back(i2);
    t.points.at(i2).lastSegmentIndex = index;
    t.points.at(i2).prevNumSegments = numSegments ;
    t.points.at(i2).prevLength = sqrt(dot(points[i2] - points[i1], points[i2] - points[i1]));
}

void trunkToSpline(std::vector<vec3f> &points, Tree &t, int &tree_index){
    int numSegments = 15;
    
    vec3f p1, cp1, cp2;

    p1 = points[tree_index];
    vec3f &p2 = points[tree_index + gridZeroPointsCount];
    p2[2] = MAX_FOREST_HEIGHT * (0.5f * (float) rand() / RAND_MAX + 0.5f);

    vec3f root_Dir = normalize(t.center - p1);
    root_Dir[2] = 0.3f * rand() / RAND_MAX + 0.7f;

    cp1 = p1 + root_Dir * 0.4f;
    cp2 = cp1 + (p2 - p1) * 0.5f;
    std::vector <vec3f> cPoints = {p1, cp1, cp2, p2};
    
    addSplineToTrees(points, t, cPoints, tree_index, tree_index+gridZeroPointsCount, numSegments);

    (*t.points.begin()).second.direction = normalize(p2 - cp2);
}



void edgeToSpline(std::vector<vec3f> &points, std::vector<Tree> &trees){

    int default_numSegments = 10;
    int count = -1;
    for (auto &t: trees){
        if (t.numBranches != -1){
            int grid_index;
            count++;
            trunkToSpline(points, t, count);

            for (int i = 0; i < t.numBranches; i++){
                    
                vec3f &p1 = points[(t.branches[i].i1)];
                vec3f &p2 = points[(t.branches[i].i2)];

            /* The higher the branching, the less the segments amount */
                grid_index = t.points.at(t.branches[i].i1).grid_index;
                int numSegments = ceil(default_numSegments * (BRANCHING - grid_index) / BRANCHING);
                if (numSegments < 3) numSegments = 3; //! BUGGY AT < 3 SEGMENTS
                
            /* Randomize a branching point from parent branch for the current point */
                int branchPointIndex;
                vec3f *branchPoint;
                int randSegment = 0;
                
                int &prevNumSegments = t.points.at(t.branches[i].i1).prevNumSegments;
                int &lastSegmentIndex = t.points.at(t.branches[i].i1).lastSegmentIndex;
                std::vector<int> &prevIndices = t.points.at(t.branches[i].i1).prevIndices;

                randSegment = rand() % ((prevNumSegments - 1)/2) + (prevNumSegments - 1)/2; // Just branching from the end half of the parent branch

                branchPointIndex = prevIndices[randSegment];
                branchPoint = &points[branchPointIndex];

            /* Branch styling */
                float prevLength = t.points.at(t.branches[i].i1).prevLength;
                float edgeLength = prevLength * (1.0f - (float)randSegment / prevNumSegments);

                float l2 = dot(*branchPoint - p2, *branchPoint - p2);
                float deltasqrd = edgeLength*edgeLength - l2;
                deltasqrd = deltasqrd < 0.1f ?  0.1f : deltasqrd;
                
                p2[2] = (*branchPoint)[2] + sqrt(deltasqrd);

            /* Pre-process edge to spline */
                vec3f direction = t.points.at(t.branches[i].i1).direction;
                
                vec3f cp1 = *branchPoint + (direction) * (1.0f / BRANCHING);
                vec3f cp2;

                if (grid_index == BRANCHING - 1){
                    cp2 = (p2 - p1) * 0.75f + p1;
                }
                else{
                    cp2 = cp1 + (p2 - *branchPoint) * 0.5f;

                }

                std::vector<vec3f> cPoints = {*branchPoint, cp1, cp2, p2};

                t.points.at(t.branches[i].i2).direction = normalize(p2 - cp2);

                addSplineToTrees(points, t, cPoints, branchPointIndex, t.branches[i].i2, numSegments);

            }
        }
    }    
}
