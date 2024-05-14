#include <iostream>
#include <fstream> 
#include "dataStructures.h"
#include <numeric>

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"
#include "../stb/stb_image_resize2.h"

#include "global_variables.h"

uint32_t coordToIndex(const Coord & c, const std::vector<Grid2D>& grids);

void branch_styling(std::vector<Grid2D> &grids, std::vector<Edge> &edges, std::vector<vec3f> &points, std::vector<int> &trees);

void write_to_OBJ(std::vector<Grid2D> grids, std::vector<Edge> edges, std::vector<vec3f> points, std::vector<int> &trees);

void filter_trees(std::vector<int> &filter_trees);

extern int gridZeroPointsCount;