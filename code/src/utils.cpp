#include "utils.h"
#include "json.hpp"
using json = nlohmann::json;

   
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

            // Writing the edges
            for (int e = 0; e < current_tree.numBranches; e++){
                
                Branch &current_branch = current_tree.branches[e];
                // Branch &current_branch = current_tree.spline_Branches[e];


                ofs << "l " << (current_branch.i1)+1 << " " << (current_branch.i2)+1 << "\n"; 
            }

            // ofs << "l " << count+1 << " " << count+1+ gridZeroPointsCount << "\n";

            ofs << "\n";
        }

    }
    ofs.close();
}

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