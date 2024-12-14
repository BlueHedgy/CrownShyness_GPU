#ifndef UTILS_H
#define UTILS_H

    #include <iostream>
    #include <iomanip>
    #include <fstream>
    #include <numeric> 

    #include "../stb/stb_image.h"
    #include "../stb/stb_image_write.h"
    #include "../stb/stb_image_resize2.h"

    #include "dataStructures_cuda.h"
    #include "global_variables.h"

// Miscellaneous 
    void load_Config_Profile(std::string filename);

    void write_to_OBJ(std::vector<vec3f> points, std::vector<Tree> &trees);

    std::vector<std::vector<float>> user_density_map(std::string filename, int subdiv);


//  Grids to points list helper functions
    extern int gridZeroPointsCount;

    uint32_t coordToIndex(const Coord & c, const std::vector<Grid2D>& grids);

    std::pair<int, Point> pointFromCoord(const Coord &c, const std::vector<Grid2D> &grids, std::vector<vec3f> &points);


// Forest control functions
    void branch_styling(std::vector<vec3f> &points, std::vector<Tree> &trees);

    void filter_trees(std::vector<Tree> &trees);

    void crownShyness(std::vector<vec3f> &points, std::vector<Tree>&trees); 

    void forest_height(std::vector<vec3f> &points, std::vector<Tree>&trees);

    void edgeToSpline(std::vector<vec3f> &points, std::vector<Tree> &trees);

#endif