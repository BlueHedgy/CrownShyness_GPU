#include <LavaCake/Math/basics.h>
#include <vector>

#include "ultils.h"


std::vector<LavaCake::vec2f> randomizeDenseCenter(int dense_region_count, int init_subdiv);

std::vector<std::vector<float>> random_density_map (int dense_region_count, int subdiv);

Grid2D generateGrid(uint16_t subdivision, int seed, int gridLayer,std::string filename, int &point_index);

std::vector<std::vector<float>> user_density_map(std::string filename, int subdiv);

Coord getClosestPoint(const Grid2D & grid, const LavaCake::vec3f & point, const uint32_t gridLayer);