#include "jitteredGrid_GPU.h"
#include <random>

using namespace LavaCake;


int tree_index = -1;
struct texture_Image{
    std::vector<std::vector<float>> weight_map;
};

__global__ void debugIndex() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

__global__ void generateCells_GPU(uint16_t init_subdiv, bool isTextureUsed, float* density_images){ 

    Grid2D_GPU grid(init_subdiv * init_subdiv);
    // cudaMalloc(&grid.cells, pow(init_subdiv, 2) * sizeof(Cell_GPU));
    // cudaMalloc(&grid.pointsCount, init_subdiv * sizeof(point_Info)); 
    int thread_Index = threadIdx.x + blockIdx.x * blockDim.x;

    printf("THREAD ID IS:  %d", thread_Index);
}

void generateGrid_GPU(uint16_t subdivision, int seed, int gridLayer, std::string filename, int &point_index){

    std::vector<Grid2D> grids(subdivision);

    srand(seed + 124534);

    //  Preload all density maps for point generation
    // std::vector<std::vector<float>> weight_maps;
    std::vector<float> weight_maps;

    for (int i = 0; i < BRANCHING; i++){
        std::vector<float> density_map;
        if (!filename.empty()){
            density_map = user_density_map(filename, subdivision);
        }

        weight_maps.insert(weight_maps.end(), density_map.begin(), density_map.end());
    }

    int nThreads = 0;
    for (int i = 0; i < BRANCHING; i++){
        nThreads += pow(subdivision * (i+1),2);
    }
    std::cout <<nThreads << std::endl;
    
    generateCells_GPU<<<1, nThreads>>>(subdivision, !filename.empty(), weight_maps.data());
    // debugIndex<<<3, 8>>>();
    cudaDeviceSynchronize();

    std::cout << "GOT HERE" << std::endl;
/*     for(uint16_t j = 0; j < subdivision ; j++){
        // std::vector<Cell> currentCellRow;
        // std::vector<int> currentCellPointsCount;

        for(uint16_t  i = 0; i < subdivision ; i++){

            int pointCount;

            // if (!filename.empty()){
            //     pointCount = int(float(MAX_POINT_PER_CELL) * weight_map[j][i]);
            // }
            // else{
            //     pointCount = MAX_POINT_PER_CELL;            
                
            // }

            pointCount = MAX_POINT_PER_CELL;
            

            Cell currentCell(pointCount);
            point_Info newPoint;
            
            for (uint16_t c = 0; c < pointCount; c++){
                point_index++;
                vec3f point;
                point [0] =  ((float)rand() / RAND_MAX + float(i)) / float(subdivision);
                point [1] =  ((float)rand() / RAND_MAX + float(j)) / float(subdivision);
                point [2] =  float(gridLayer);
            
                // currentCell.points.push_back(point);
                currentCell.points[c] = point;
           
                float weight;
                // Randomized weight for testing
                if (gridLayer == 0){
                    newPoint.points_weight  = (((float)rand() / (RAND_MAX)) + 0.01) * 1.0 / (1.4 * INIT_SUBDIV);
                    tree_index++;
                    newPoint.tree_index = tree_index;
                    newPoint.strength = 1.0f;
                }
                else{
                    weight = 1.0f;
                    newPoint.tree_index = -1;


                }

                newPoint.global_point_index = point_index;    

                // currentCell.pointsInfo.push_back(newPoint);
                currentCell.pointsInfo[c] = newPoint;

            }
            grid.cells[j * subdivision + i] = currentCell;
            grid.pointsCount[j * subdivision + i] = pointCount;
        }


    }
     */
}


// Coord getClosestPoint(const Grid2D& grid, const vec3f& point, const  uint32_t gridLayer){

//     // get the corresponding cell in the lower level projected from the current point
//     vec2i cell({int(point[0] * grid.cells[0].size() ),int(point[1] * grid.cells.size())});
//     Coord closestPoint;
//     float mindistsqrd = 10.0f;

//     for(int j = cell[1]-1; j <= cell[1] +1 ; j++){
//         for(int i = cell[0]-1; i <=cell[0]+1 ; i++){
//             // make sure the cells that are being checked is within the boundary of the grids
//             if( i >= 0 &&  i < grid.cells[0].size() &&  j >= 0 && j < grid.cells.size()){

//                 const Cell &currentCell = (grid.cells[j][i]);
//                 for (int p = 0; p < grid.cells[j][i].points.size(); p++){
                    
//                     // d² 
//                     float distsqrd  = dot(point - currentCell.points[p],  point - currentCell.points[p]);

//                     float power = distsqrd - pow(currentCell.pointsInfo[p].points_weight, 2.0);

//                     if(power < mindistsqrd){
//                         mindistsqrd = power;
 
//                         closestPoint.gridIndex = gridLayer;
//                         closestPoint.coord[0] = i;
//                         closestPoint.coord[1] = j;
//                         closestPoint.pointIndex = p;
//                         closestPoint.weight = currentCell.pointsInfo[p].points_weight;
//                         closestPoint.tree_index = currentCell.pointsInfo[p].tree_index;
//                     }
//                 }
//             }
//         }
//     }

//     return closestPoint;

// }