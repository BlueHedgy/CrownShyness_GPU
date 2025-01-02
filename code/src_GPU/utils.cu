#include "utils_GPU.h"
#include "json.hpp"
using json = nlohmann::json;

// DEFAULT CONFIG PARAMETERS
int BRANCHING;                                  // Number of grid layers
int INIT_SUBDIV;                                // Layer 0 subdivisions
float GEN_AREA;                                 // Generation area
float SCALE;                                    // Scale value used in some ops 
int MAX_POINT_PER_CELL;                         // Max amount of points per grid cell 
float WEIGHT_ATTENUATION;                       // Root points have weight, diminished by layer index
std::string DENSITY_IMAGE;                      // Input density image for generation land
std::string SHRINK_FACTOR_IMAGE;                // Control how much trees of area shrinks
int CROWN_SHYNESS_STEP;                         // How many times the shrink effect happens across the layers, <= BRANCHING
bool BRANCH_STYLING;                            // Heuristics tree branches' length function for the straight edges
bool FILTER_TREES;
int BRANCHES_COUNT_THRESHOLD;                   // Lower limit to how dense tree needs to be to be kept in output
float DEFAULT_SHRINK_FACTOR;                    // Shrink coefficient for crownshyness effect
float MAX_FOREST_HEIGHT;
float MIN_FOREST_HEIGHT;
std::string FOREST_HEIGHT_IMAGE;

void load_Config_Profile(std::string filename){
    std::string profile_name;

    if (filename != ""){
        std::ifstream configFile(filename);
        if (!configFile.is_open()){
            std::cerr << "FAILED TO LOAD CONFIGURATION FILE" << std::endl;
            return;
        }

        json config = json::parse(configFile);
        configFile.close();

        std::cout << "Enter configuration profile " << "\n";
        std::cout << "Leave blank for default configs: \n"; 
        do{
            std::cout << "> ";
            std::getline(std::cin, profile_name);
            if (profile_name.empty()){
                std::cout << "Using default configurations" << std::endl;
                profile_name = "default";
            }

            else if (!config.contains(profile_name)){
                std::cout << "Profile: " << profile_name << " does not exists !"<< "\n";
            }


        }
        while(!profile_name.empty() && !config.contains(profile_name));
        std::cout << "\n";


        BRANCHING                       = config.at(profile_name).at("BRANCHING");
        INIT_SUBDIV                     = config.at(profile_name).at("INIT_SUBDIV");
        GEN_AREA                        = config.at(profile_name).at("GEN_AREA");
        SCALE                           = config.at(profile_name).at("SCALE");
        MAX_POINT_PER_CELL              = config.at(profile_name).at("MAX_POINT_PER_CELL");
        WEIGHT_ATTENUATION              = config.at(profile_name).at("WEIGHT_ATTENUATION");
        DENSITY_IMAGE                   = config.at(profile_name).at("DENSITY_IMAGE");
        SHRINK_FACTOR_IMAGE             = config.at(profile_name).at("SHRINK_FACTOR_IMAGE");
        CROWN_SHYNESS_STEP              = config.at(profile_name).at("CROWN_SHYNESS_STEP");
        BRANCH_STYLING                  = config.at(profile_name).at("BRANCH_STYLING");
        FILTER_TREES                    = config.at(profile_name).at("FILTER_TREES");
        BRANCHES_COUNT_THRESHOLD        = config.at(profile_name).at("BRANCHES_COUNT_THRESHOLD");
        DEFAULT_SHRINK_FACTOR           = config.at(profile_name).at("DEFAULT_SHRINK_FACTOR");
        MAX_FOREST_HEIGHT               = config.at(profile_name).at("MAX_FOREST_HEIGHT");
        MIN_FOREST_HEIGHT               = config.at(profile_name).at("MIN_FOREST_HEIGHT");
        FOREST_HEIGHT_IMAGE             = config.at(profile_name).at("FOREST_HEIGHT_IMAGE");

    }
    else{
        std::cout << "No configuration files, using default parameters" << std::endl;
    }
}


// DENSITY --------------------------------------------------------------

std::vector<std::vector<float>> user_density_map(std::string filename, int subdiv){
    int width, height, channelsNum;
    int desiredChannels = 1; // grayscale

    stbi_set_flip_vertically_on_load(true);
    unsigned char * image = stbi_load(filename.c_str(), &width, &height, &channelsNum, desiredChannels);

    if (image == NULL){
        std::cout << "Failed to load density map\n" << std::endl;
        exit(1);
    }

    unsigned char * resized_im;
    if (subdiv != 0){
        resized_im = stbir_resize_uint8_srgb(image, width, height, 0, NULL, subdiv, subdiv, 0, STBIR_1CHANNEL);
        height = subdiv;
        width = subdiv;

        stbi_image_free(image);
    }
    else{
        resized_im = image;
    }

    std::vector<std::vector<float>> map;

    for (int j = 0; j < height; j++){
        std::vector<float> currentRow;
        for (int i = 0; i < width; i++){
            currentRow.push_back(1.0 - float(int(resized_im[j * width + i])/255.0));
        }
        map.push_back(currentRow);
        
    }
    stbi_image_free(resized_im);

    return map;
}

std::vector<float> user_density_map_flat(std::string filename, int subdiv){
    int width, height, channelsNum;
    int desiredChannels = 1; // grayscale

    stbi_set_flip_vertically_on_load(true);
    unsigned char * image = stbi_load(filename.c_str(), &width, &height, &channelsNum, desiredChannels);

    if (image == NULL){
        std::cout << "Failed to load density map\n" << std::endl;
        exit(1);
    }

    unsigned char * resized_im;
    if (subdiv != 0){
        resized_im = stbir_resize_uint8_srgb(image, width, height, 0, NULL, subdiv, subdiv, 0, STBIR_1CHANNEL);
        height = subdiv;
        width = subdiv;

        stbi_image_free(image);
    }
    else{
        resized_im = image;
    }

    // std::vector<std::vector<float>> map;
    std::vector<float> map;

    for (int j = 0; j < height; j++){
        for (int i = 0; i < width; i++){
            map.push_back(1.0 - float(int(resized_im[j * width + i])/255.0));
        }
    }
    stbi_image_free(resized_im);

    return map;
}


void write_to_OBJ(std::vector<vec3f> points, std::vector<Tree> &trees){
    // Write to OBJ
    std::ofstream ofs;
    ofs.open("graph.obj", std::ofstream::out | std::ofstream::trunc);

    for (int k = 0; k < points.size(); k++) {
        ofs << std::fixed << std::setprecision(4) << "v " << points[k][0] << " " << points[k][1] << " " <<  points[k][2]  << "\n";
    }

    int count = -1;
    for (int i = 0; i < trees.size(); i++){
        Tree &current_tree = trees[i];
    
        if (current_tree.numBranches != -1){
            count++;
            
            ofs << "o " << "Tree_"<< std::to_string(trees[i].ID) << "\n";

        // Writing the curved branches
            for (int e = 0; e < current_tree.numSplineBranches; e++){
                
                Branch &current_branch = current_tree.spline_Branches[e];

                ofs << "l " << (current_branch.i1)+1 << " " << (current_branch.i2)+1 << "\n"; 
            }

        // Replace the loop above with this segments to switch back to straight edges tree
        // Also enable the Branch styling and comment out the edgetoSpline in main.cpp
            // for (int e = 0; e < current_tree.numBranches; e++){
                
            //     Branch &current_branch = current_tree.branches[e];

            //     ofs << "l " << (current_branch.i1)+1 << " " << (current_branch.i2)+1 << "\n"; 
            // }

            // ofs << "l " << count+1 << " " << count+1+ gridZeroPointsCount << "\n";

            // ofs << "\n";
       

        }
    }
    ofs.close();
}
