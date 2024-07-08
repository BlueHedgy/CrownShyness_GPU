#include "utils.h"
#include "json.hpp"
using json = nlohmann::json;


vec3f lerp (vec3f &p1, vec3f &p2, float t){
    const float s = 1.0 - t;
    return vec3f ({ p1[0] * s + p2[0] * t, 
                    p1[1] * s + p2[1] * t, 
                    p1[2] * s + p2[2] * t
                  });
}

vec3f De_Casteljau_Algo(std::vector<vec3f> cPoints, float segment_coeff){
    if (cPoints.size() > 1){
        std::vector<vec3f> new_cPoints;
        for (int p = 0; p < cPoints.size() - 1; p++){
            vec3f new_cPoint = lerp(cPoints[p], cPoints[p+1], segment_coeff);
            new_cPoints.push_back(new_cPoint);

        }

        return De_Casteljau_Algo(new_cPoints, segment_coeff);
    }

    // std::cout << cPoints[0][0] << " " << cPoints[0][1] << " " << cPoints[0][2] << std::endl;
    return cPoints[0];

}

void edgeToSpline(std::vector<vec3f> &points, std::vector<Tree> &trees){
    // std::vector<std::pair

    int numSegments = 3;

    for (auto &t: trees){
        if (t.numBranches != -1){
            int splineBranches = 0;
            int grid_index = 0;
            for (int i = 0; i < t.numBranches; i++){

                vec3f &prevDirection = t.points.at(t.branches[i].i1).direction;

                vec3f p1 = points[(t.branches[i].i1)];
                vec3f p2 = points[(t.branches[i].i2)];
                vec3f cp1 = p1 + prevDirection / vec3f({5.0, 5.0, 5.0});
                vec3f cp2 = cp1 + Normalize(p2 - p1)/ vec3f({20.0, 20.0, 20.0});

                std::vector<vec3f> cPoints = {p1, cp1, cp2, p2};
                // std::vector<vec3f> cPoints = {p2, cp2, cp1, p1};

                for (int s = 1; s < numSegments; s++){
                    float coeff = ((float)s)/numSegments;
                    vec3f pt = De_Casteljau_Algo(cPoints, coeff);

                    points.push_back(pt);
                    int index = points.size() -1;
                    if (s == 1) {
                        t.spline_Branches.push_back(Branch({t.branches[i].i1, index}));
                    }
                    else if (s == numSegments-1){
                        t.spline_Branches.push_back(Branch({index, t.branches[i].i2}));
                        t.spline_Branches.push_back(Branch({index-1, index}));
                        splineBranches++;
                    }
                    else{
                        t.spline_Branches.push_back(Branch({index-1, index}));

                    }
                    splineBranches++;
                }

            }
            t.numBranches = splineBranches;
        }
    }    
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

            // Writing the edges
            for (int e = 0; e < current_tree.numBranches; e++){
                
                // Branch &current_branch = current_tree.branches[e];
                Branch &current_branch = current_tree.spline_Branches[e];


                ofs << "l " << (current_branch.i1)+1 << " " << (current_branch.i2)+1 << "\n"; 
            }

            ofs << "l " << count+1 << " " << count+1+ gridZeroPointsCount << "\n";

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