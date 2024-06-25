#ifndef GLOBAL_VARIABLES_H
#define GLOBAL_VARIABLES_H
    #define STRING(x) #x            // Don't touch this
    #define XSTRING(x) STRING(x)    // Nor this

    extern int BRANCHING;
    extern int INIT_SUBDIV;
    extern float GEN_AREA;
    extern float SCALE;
    extern int MAX_POINT_PER_CELL;
    extern float WEIGHT_ATTENUATION;
    extern std::string DENSITY_IMAGE;
    extern std::string SHRINK_FACTOR_IMAGE;
    extern int CROWN_SHYNESS_STEP;
    extern bool BRANCH_STYLING;
    extern bool FILTER_TREES;
    extern int BRANCHES_COUNT_THRESHOLD;
    extern float DEFAULT_SHRINK_FACTOR;
    extern float MAX_FOREST_HEIGHT;
    extern float MIN_FOREST_HEIGHT;
    extern std::string FOREST_HEIGHT_IMAGE;

#endif