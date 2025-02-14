#include "jitteredGrid_GPU.h"
#include <random>
#include <curand_kernel.h>
#include <math.h>

using namespace LavaCake;


int tree_index = -1;
struct texture_Image{
    std::vector<std::vector<float>> weight_map;
};

// __global__ void debugIndex() {
//     int threadId = threadIdx.x + blockIdx.x * blockDim.x;
//     printf("Thread %d in block %d\n", threadIdx.x, blockIdx.x);
// }


__device__ void test(Cell_GPU *cells, generationInfo *genInfo){
    printf("TEST SUCCESSFUL! \n");
}

__global__ void edgeConnection(Cell_GPU * cells, generationInfo *genInfo){
    int thread_Index = threadIdx.x + blockIdx.x * blockDim.x;
    
    int gridLayer = 0;
    for (int l = 0; l < genInfo->branching; l++){
        if (thread_Index < genInfo->layer_MileStones[l]){
            gridLayer = l;
            break;
        }
    }
    
    if (gridLayer > 0){
        // printf("HELLO %d\n", gridLayer);
        Cell_GPU *currentCell = &cells[thread_Index];
        
        int layerSubdiv = genInfo->init_subdiv * powf(genInfo->scale, gridLayer);
        int prevLayerSubdiv = genInfo->init_subdiv * powf(genInfo->scale, gridLayer-1);

        // printf("%d %d\n", layerSubdiv, prevLayerSubdiv);
        // printf("%d\n", genInfo->scale);
        int search_area = 3;
        
        int point_count = genInfo->MAX_POINT_PER_CELL;
        if (genInfo->isTextureUsed) point_count = genInfo->density_images[thread_Index] * (genInfo->MAX_POINT_PER_CELL);
        
        int projected_ThreadIdx = 0;
        
        for (int p = 0; p < point_count; p++){
            float x = currentCell->points[p*3];
            float y = currentCell->points[p*3+1];
            float z = currentCell->points[p*3+2];
            
            
            projected_ThreadIdx = x * prevLayerSubdiv + y * prevLayerSubdiv * prevLayerSubdiv + genInfo->layer_MileStones[gridLayer - 1];
            
            int currentPointIndex = currentCell->pointsInfo[p].global_point_index;

            int closest_PointIndex = 0;

            float mindistsqrd = 10.0;

            for (int i = -1; i < 2; i++){   
                for (int j = -1; j < 2; j++){
                    int neighborIndex = projected_ThreadIdx + i * layerSubdiv  + j;
                    if ( genInfo->layer_MileStones[gridLayer -1] < neighborIndex && neighborIndex < genInfo->layer_MileStones[gridLayer]){
                        
                        Cell_GPU * neighborCell = &cells[neighborIndex];
                        int point_count_n = genInfo->MAX_POINT_PER_CELL;
                        if (genInfo->isTextureUsed) point_count_n = genInfo->density_images[projected_ThreadIdx] * (genInfo->MAX_POINT_PER_CELL);
            
                        for (int p1 = 0; p1 < point_count_n; p1++){
                            float x1 = neighborCell->points[p1*3];
                            float y1 = neighborCell->points[p1*3 + 1];
            
                            float distsqrd = pow(x1 - x, 2.0) + pow(y1 - y, 2.0);
                            float power = distsqrd - pow(neighborCell->pointsInfo[p1].points_weight, 2.0);
                          
                            if (power < mindistsqrd){
                                mindistsqrd = power;
                                
                                closest_PointIndex = neighborCell->pointsInfo[p1].global_point_index; 
                            }
                        }   
                    }
        
                }
            }

            // genInfo->edges[currentPointIndex].p1 = currentPointIndex;
            // genInfo->edges[currentPointIndex].p2 = closest_PointIndex;



        //     // printf("TID: %d ... %f %f %f\n", thread_Index, cells[thread_Index].points[p*3], cells[thread_Index].points[p*3+1], cells[thread_Index].points[p*3+2]); 
            
        }
    }
    
}



// __global__ void generateCells_GPU(uint16_t* init_subdiv, bool* isTextureUsed, float* density_images, uint16_t* d_MAX_POINT_PER_CELL, Cell_GPU* cells, int* nCellsThread, int* layer_Milestones){ 
__global__ void generateCells_GPU(generationInfo *genInfo, Cell_GPU* cells){ 

    int thread_Index = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_Index < genInfo->nCellsThread){
        int point_count = genInfo->MAX_POINT_PER_CELL;
        if (genInfo->isTextureUsed) point_count = genInfo->density_images[thread_Index] * (genInfo->MAX_POINT_PER_CELL);
        

        curandState state;
        curand_init(393452, threadIdx.x, 0, &state);

        int gridLayer;

        for (int l = 0; l < genInfo->branching; l++){
            if (thread_Index < genInfo->layer_MileStones[l]){
                gridLayer = l;
                break;
            }
        }
                
        Cell_GPU *currentCell = &cells[thread_Index];
        
        currentCell->point_count = point_count;

        for (int p = 0; p < point_count; p++){
            currentCell->points[p*3] = curand_uniform(&state);
            currentCell->points[p*3+1] = curand_uniform(&state);
            currentCell->points[p*3+2] = gridLayer;

            
            if (gridLayer == 0){
                currentCell->pointsInfo[p].points_weight = curand_uniform(&state) / (genInfo->init_subdiv);
                currentCell->pointsInfo[p].tree_index = thread_Index;
            }
            else{
                currentCell->pointsInfo[p].points_weight = 1.0;
                currentCell->pointsInfo[p].tree_index = -1;
            }

            currentCell->pointsInfo[p].global_point_index = thread_Index * genInfo->MAX_POINT_PER_CELL + p;

        }   

        
        
        // if (thread_Index == 0){
        // for (int p = 0; p < point_count; p++){
        //     printf("TID: %d ... %f %f %f\n", thread_Index, cells[thread_Index].points[p*3], cells[thread_Index].points[p*3+1], cells[thread_Index].points[p*3+2]); 
        // }
        // }

        // printf("%d\n", gridLayer);
    }
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
    uint16_t blockSize = 256;
    int nBlocks = nCellThreads / blockSize + 1 ;
    

    // ALLOCATING CUDA GRID CELLS VARIABLES
    
    generationInfo h_genInfo;
    h_genInfo.init_subdiv = INIT_SUBDIV;
    h_genInfo.isTextureUsed = h_isTextureUsed;
    h_genInfo.MAX_POINT_PER_CELL = MAX_POINT_PER_CELL;
    h_genInfo.scale = SCALE;
    h_genInfo.branching = BRANCHING;
    h_genInfo.nCellsThread = nCellThreads;
    
    float *d_density_images; 
    int *d_layer_MileStones;
    Edge_GPU *d_edges;
    
    generationInfo *d_genInfo;
    
    
    Cell_GPU *d_Cells;
    
    // ---------------------------------------------------------------------------------------------------------
    cudaMalloc(&d_layer_MileStones, BRANCHING * sizeof(int));
    cudaMemcpy(d_layer_MileStones, layer_Milestones.data(), BRANCHING*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_density_images, sizeof(float)*weight_maps.size());
    cudaMemcpy(d_density_images, weight_maps.data(), sizeof(float)*weight_maps.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_edges, sizeof(Edge_GPU) * layer_Milestones[BRANCHING-1]);
    h_genInfo.density_images = d_density_images;
    h_genInfo.layer_MileStones = d_layer_MileStones;
    h_genInfo.edges = d_edges;
    
    cudaMalloc(&d_genInfo, sizeof(h_genInfo));
    cudaMemcpy(d_genInfo, &h_genInfo, sizeof(h_genInfo), cudaMemcpyHostToDevice);
    
    
    //---------------------------------------------------------------------------------------------------------------
    
    Cell_GPU * h_cells = new Cell_GPU[nCellThreads];
    cudaMalloc(&d_Cells, nCellThreads * sizeof(Cell_GPU));
    
    for (int i = 0; i < nCellThreads; i++){
        int numPoints = MAX_POINT_PER_CELL;
        if (h_isTextureUsed) int numPoints = weight_maps[i] * MAX_POINT_PER_CELL;
        
        float *d_points = nullptr;
        point_Info *d_pointsInfo = nullptr;

        cudaMalloc((void**)&d_points, numPoints * 3 * sizeof(float));
        cudaMalloc((void**)&d_pointsInfo, numPoints * sizeof(float));

        h_cells[i].points = d_points;   //store the pointer to device-points-array inside in host-points-array
        h_cells[i].pointsInfo = d_pointsInfo;
        
    }
    
    // copy h_cells to d_cells with d_cells now contains the pointers to pre allocated inner structures and array
    cudaMemcpy(d_Cells, h_cells, nCellThreads * sizeof(Cell_GPU), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    generateCells_GPU<<<nBlocks, blockSize>>>(d_genInfo, d_Cells);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    edgeConnection<<<nBlocks, blockSize>>>(d_Cells, d_genInfo);
    cudaDeviceSynchronize();

    
    cudaEventRecord(stop);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "CUDA RUN TIME: " << milliseconds << std::endl;


    
    // float **d_points_array = new float*[nCellThreads]; // array of pointers to all the arrays of float on device
    // for (int i = 0; i < nCellThreads; i++){
    //     d_points_array[i] = h_cells[i].points; // copied from previously saved device pointer value
    // }
    
    // for (int i = 0; i < nCellThreads; i++){
    //     int numPoints = MAX_POINT_PER_CELL;
    //     if (h_isTextureUsed) int numPoints = weight_maps[i] * MAX_POINT_PER_CELL;

    //     float *h_currCellPoints = new float[numPoints * 3];
        

    //     cudaMemcpy(h_currCellPoints, h_cells[i].points, numPoints * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    //     h_cells[i].points = h_currCellPoints;
    // }


    // std::cout << std::fixed << std::setprecision(6) << h_cells[0].points[0*3] << " " << h_cells[0].points[0*3 + 1] << " " << h_cells[0].points[0*3 + 2] << std::endl;

    
    std::cout << "GOT HERE 1" << std::endl;


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