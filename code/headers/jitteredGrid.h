#include <LavaCake/Math/basics.h>
#include <vector>

#include "utils.h"

// CPU
Grid2D generateGrid(uint16_t subdivision, int seed, int gridLayer,std::string filename, int &point_index);

Coord getClosestPoint(const Grid2D & grid, const LavaCake::vec3f & point, const uint32_t gridLayer);

// GPU
Grid2D_p generateGrid_p(uint16_t subdivision, int seed, int gridLayer,std::string filename, int &point_index);

Coord_p getClosestPoint_p(const Grid2D & grid, const LavaCake::vec3f & point, const uint32_t gridLayer);