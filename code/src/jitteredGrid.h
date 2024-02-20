#include <LavaCake/Math/basics.h>
#include <vector>

#include "dataStructures.h"

// struct Grid2D{
//     std::vector<std::vector<LavaCake::vec2f>> points;
//     bool isDenseCenter;
// };

Grid2D generateGrid(u_int16_t subdivision, int seed);


Coord getClosestPoint(const Grid2D & grid, const LavaCake::vec2f & point, const  uint32_t gridLayer);