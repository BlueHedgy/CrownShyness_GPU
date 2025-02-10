#include "jitteredGrid_GPU.h"
#include <random>
#include <curand_kernel.h>

using namespace LavaCake;


int tree_index = -1;
struct texture_Image{
    std::vector<std::vector<float>> weight_map;
};

// __global__ void debugIndex() {
//     int threadId = threadIdx.x + blockIdx.x * blockDim.x;
//     printf("Thread %d in block %d\n", threadIdx.x, blockIdx.x);
// }

__global__ void generateCells_GPU(uint16_t* init_subdiv, bool* isTextureUsed, float* density_images, uint16_t* d_MAX_POINT_PER_CELL, Cell_GPU* cells){ 

    int thread_Index = threadIdx.x + blockIdx.x * blockDim.x;
    int point_count = *d_MAX_POINT_PER_CELL;
    if (*isTextureUsed) point_count = density_images[thread_Index] * (*d_MAX_POINT_PER_CELL);

    curandState state;
    curand_init(393452, threadIdx.x, 0, &state);

    
    Cell_GPU currentCell(point_count);
    
    for (int p = 0; p < point_count; p++){
        currentCell.points[p*3] = curand_uniform(&state);
        currentCell.points[p*3+1] = curand_uniform(&state);
        currentCell.points[p*3+2] = curand_uniform(&state);
    }

    cells[thread_Index] = currentCell;  // no race condition here, each thead (Cell) updates independently
    
    // for (int p = 0; p < point_count; p++){
    //     printf("%f %f %f\n", cells[thread_Index].points[p], cells[thread_Index].points[p+1], cells[thread_Index].points[p+2]); 
    // }
}

void generateGrid_GPU(uint16_t  subdivision, int seed, int gridLayer, std::string filename, int &point_index){


    srand(seed + 124534);

    //  Preload all density maps for point generation
    std::vector<float> weight_maps;

    // Precompute the amount of cells on each layer 
    std::vector<int> layer_Milestones(BRANCHING);

    // Load the user defined texture image forr density
    for (int i = 0; i < BRANCHING; i++){
        std::vector<float> density_map;
        if (!filename.empty()){
            density_map = user_density_map_flat(filename, subdivision);
        }
        for (int j = 0; j < density_map.size(); j++){
            weight_maps.push_back(density_map[j]);
        }
        
        subdivision *= SCALE;
    }

    // Compute number of threads == number of cells
    int nCellThreads = 0;
    int init_subdiv = INIT_SUBDIV;

    for (int i = 0; i < BRANCHING; i++){
        nCellThreads += pow(init_subdiv ,2);
        init_subdiv *= SCALE;
        
        // Store the milestones for cell numbers, this helps with the 1D grid array on CUDA
        layer_Milestones[i] = nCellThreads;
    }

    printf("TOTAL NUMBER OF CELLS: %d\n", nCellThreads);
    
    // Init variables for CUDA segment
    bool h_isTextureUsed = !filename.empty();
    int nBlocks = nCellThreads / 1024 + 1 ;
    uint16_t blockSize = 1024;
    

    // ALLOCATING CUDA GRID CELLS VARIABLES
    uint16_t *d_subdiv;
    bool *d_isTextureUsed;
    float *d_density_images; 
    uint16_t *d_MAX_POINT_PER_CELL;
    Cell_GPU *d_Cells;
    Grid2D_GPU * d_grids;

    int* d_layer_MileStones;
    // float* d_points_list;
    
    // cudaMalloc(&d_grids, BRANCHING*sizeof(Grid2D_GPU));
    cudaMalloc(&d_Cells, nCellThreads * sizeof(Cell_GPU));
    
    cudaMalloc(&d_layer_MileStones, BRANCHING * sizeof(int));
    cudaMemcpy(d_layer_MileStones, layer_Milestones.data(), BRANCHING*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_subdiv, sizeof(uint16_t));
    cudaMemcpy(d_subdiv, &subdivision, sizeof(uint16_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_isTextureUsed, sizeof(bool));
    cudaMemcpy(d_isTextureUsed, &h_isTextureUsed, sizeof(bool), cudaMemcpyHostToDevice);

    cudaMalloc(&d_density_images, sizeof(float)*weight_maps.size());
    cudaMemcpy(d_density_images, weight_maps.data(), sizeof(float)*weight_maps.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_MAX_POINT_PER_CELL, sizeof(uint16_t));
    cudaMemcpy(d_MAX_POINT_PER_CELL, &MAX_POINT_PER_CELL, sizeof(uint16_t), cudaMemcpyHostToDevice);

    generateCells_GPU<<<nBlocks, blockSize>>>(d_subdiv, d_isTextureUsed, d_density_images, d_MAX_POINT_PER_CELL, d_Cells);
    cudaDeviceSynchronize();

    // std::cout <<  nCellThreads << " " << weight_maps.size() << std::endl;
    std::vector<Cell_GPU> h_cells(nCellThreads);
    // THIS IS NOT WORKING IDK WHY

    std::cout << "GOT HERE" << std::endl;
    // for (int i = 0; i < nCellThreads; i++){
    //     int numPoints = MAX_POINT_PER_CELL;
    //     if (h_isTextureUsed) int numPoints = weight_maps[i] * MAX_POINT_PER_CELL;

    //     h_cells[i].points.resize(3*numPoints);
    //     // std::cout << h_cells[i].points.size() << std::endl;

    //     cudaMemcpy(h_cells[i].points.data(), d_Cells[i].points, sizeof(float) * 3 * numPoints, cudaMemcpyDeviceToHost);

    //     std::cout << h_cells[i].points[0] << std::endl;
    // }

    cudaMemcpy(h_cells.data(), d_Cells, sizeof(Cell_GPU) * nCellThreads, cudaMemcpyDeviceToHost);
 
    
    std::cout << "GOT HERE 1" << std::endl;

    for (int i = 0; i <= h_cells.size(); i++){
        std::cout << h_cells[i].points[0] << std::endl;
    }

    // for (int i = 0; i < nCellThreads; i++){
    //     std::cout << "HELLO:  " << h_cells[i].points[0] <<std::endl;
    // }

    std::cout << "GOT HERE 2" << std::endl;

/*     for(uint16_t j = 0; j < subdivision ; j++){
        // std::vector<Cell> currentCellRow;
        // std::vector<int> currentCellPointsCount;

        for(uint16_t  i = 0; i < subdivision ; i++){

            int pointCount;

            if (!filename.empty()){ 
                pointCount = int(float(MAX_POINT_PER_CELL) * weight_map[j][i]);
            }
            else{
            
                pointCount = MAX_POINT_PER_CELL;            
                
            }

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