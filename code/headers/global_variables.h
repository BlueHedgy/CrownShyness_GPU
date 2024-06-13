#ifndef GLOBAL_VARIABLES
#define GLOBAL_VARIABLES
    #define STRING(x) #x            // Don't touch this
    #define XSTRING(x) STRING(x)    // Nor this

    #define BRANCHING 5             // Number of grid layers
    #define INIT_SUBDIV 18.0         // Indicate the subdivision size of the current space
                                    // i.e tile the space into 10x10 grid

    #define GEN_AREA 8.0            // area of generation (e.g 2 means 2 m^2)
                                    // recommended to be equal to INIT_SUBDIV

    #define FLATNESS 2.0            // flat constant for certain scaling operations
    #define MAX_POINT_PER_CELL 6.0  // Self-explanatory

    #define WEIGHT_ATTENUATION 1.0/2.0

    #define DENSITY_IMAGE XSTRING(CMAKE_SOURCE_DIR)"/test_images/Density1.png"
    // #define DENSITY_IMAGE ""
    
    #define CROWN_SHYNESS_STEP 1.0
    #define BRANCH_STYLING true     // Set this to false for debugging, not recommended

    #define FILTER_TREES true       // whether to remove the trees with less than 
                                    // BRANCHES_COUNT_THRESHOLD brances

    #define BRANCHES_COUNT_THRESHOLD 100

    // #define SHRINK_FACTOR_IMAGE XSTRING(CMAKE_SOURCE_DIR)"/shrink_map.png"
    #define SHRINK_FACTOR_IMAGE ""

    #define DEFAULT_TREE_SHRINK_FACTOR 0.95

#endif