#include <LavaCake/Math/basics.h>
#include <vector>


struct Grid2D{
    std::vector<std::vector<LavaCake::vec2f>> points;
};

/* struct Grid2D{
    std::vector<std::vector<LavaCake::vec2f>> points;
    bool isDenseCenter;
}; */

Grid2D generateGrid(u_int16_t subdivision, int seed);


LavaCake::vec2u closestPoint(const Grid2D & grid, const LavaCake::vec2f & point);