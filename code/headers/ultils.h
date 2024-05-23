#ifndef ULTILS_H
#define ULTILS_H

    #include <iostream>
    #include <iomanip>
    #include <fstream> 
    #include "dataStructures.h"

    #include "../stb/stb_image.h"
    #include "../stb/stb_image_write.h"
    #include "../stb/stb_image_resize2.h"

    #include "global_variables.h"

    std::pair<int, Point> pointFromCoord(const Coord &c, const std::vector<Grid2D> &grids);
    uint32_t coordToIndex(const Coord & c, const std::vector<Grid2D>& grids);

    void branch_styling(std::vector<vec3f> &points, std::vector<Tree> &trees);

    void write_to_OBJ(std::vector<vec3f> points, std::vector<Tree> &trees);

    void filter_trees(std::vector<Tree> &trees);

    std::vector<std::vector<float>> user_density_map(std::string filename, int subdiv);

    void crownShyness(std::vector<vec3f> &points, std::vector<Tree>&trees); 

    extern int gridZeroPointsCount;

#endif