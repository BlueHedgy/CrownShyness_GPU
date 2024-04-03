#include <iostream>
#include <fstream> 
#include "dataStructures.h"
#include <numeric>

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"
#include "../stb/stb_image_resize2.h"

#include "global_variables.h"

uint32_t coordToIndex(const Coord & c, const std::vector<Grid2D>& grids);

void branch_styling(std::vector<Grid2D> *grids, std::vector<Edge> *edges, std::vector<vec3f> *points);

void write_to_OBJ(std::vector<Grid2D> grids, std::vector<Edge> edges, std::vector<vec3f> points);

