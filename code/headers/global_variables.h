#ifndef GLOBAL_VARIABLES
#define GLOBAL_VARIABLES
    #define STRING(x) #x            // Don't touch this
    #define XSTRING(x) STRING(x)    // Nor this

    #define BRANCHING 5             // Number of grid layers
    #define INIT_SUBDIV 4.0         // Indicate the subdivision size of the current space
                                    // i.e tile the space into 10x10 grid

    #define GEN_AREA 4.0            // area of generation (e.g 2 means 2 m^2)

    #define FLATNESS 2.0            // flat constant for certain scaling operations
    #define MAX_POINT_PER_CELL 8.0  // Self-explanatory

    #define WEIGHT_ATTENUATION 1.0/1.5

    // #define DENSITY_IMAGE XSTRING(CMAKE_SOURCE_DIR)"/testing4.png"
    #define DENSITY_IMAGE ""

    #define BRANCH_STYLING true
    #define FILTER_TREES true
#endif