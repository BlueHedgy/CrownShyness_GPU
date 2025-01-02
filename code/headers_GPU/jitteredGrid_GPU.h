#include <LavaCake/Math/basics.h>
#include <vector>

#include "utils_GPU.h"

// Grid2D generateGrid(uint16_t subdivision, int seed, int gridLayer,std::string filename, int &point_index);

// Coord getClosestPoint(const Grid2D & grid, const LavaCake::vec3f & point, const uint32_t gridLayer);

void generateGrid_GPU(uint16_t subdivision, int seed, int gridLayer,std::string filename, int &point_index);

__global__ void generateCells_GPU(uint16_t init_subdiv, bool isTextureUsed, float* density_images);
