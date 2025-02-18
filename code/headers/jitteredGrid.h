#include <LavaCake/Math/basics.h>
#include <vector>

#include "utils.h"

Grid2D generateGrid(uint16_t subdivision, int seed, int gridLayer,std::string filename, int &point_index);

Coord getClosestPoint(const Grid2D & grid, const LavaCake::vec3f & point, const uint32_t gridLayer);
