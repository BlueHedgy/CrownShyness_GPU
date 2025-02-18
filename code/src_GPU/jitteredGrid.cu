#include "jitteredGrid_GPU.h"
#include <random>
#include <curand_kernel.h>
#include <math.h>
#include "chrono"


using namespace LavaCake;

# define BLOCKSIZE 512

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
        Cell_GPU *currentCell = &cells[thread_Index];
        
        int layerSubdiv = genInfo->init_subdiv * powf(genInfo->scale, gridLayer);
        int prevLayerSubdiv = genInfo->init_subdiv * powf(genInfo->scale, gridLayer-1);

        int search_area = 3;
        
        int point_count = currentCell->point_count;

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
                    int neighborIndex = projected_ThreadIdx + i * prevLayerSubdiv  + j;
                    if ( genInfo->layer_MileStones[gridLayer -1] < neighborIndex && neighborIndex < genInfo->layer_MileStones[gridLayer]){
                        Cell_GPU * neighborCell = &cells[neighborIndex];
                        
                        int point_count_n = neighborCell->point_count;
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
            
            
            genInfo->edges[currentPointIndex].p1 = currentPointIndex;
            genInfo->edges[currentPointIndex].p2 = closest_PointIndex;
            
        }
    }
    
}



// __global__ void generateCells_GPU(uint16_t* init_subdiv, bool* isTextureUsed, float* density_images, uint16_t* d_MAX_POINT_PER_CELL, Cell_GPU* cells, int* nCellsThread, int* layer_Milestones){ 
__global__ void generateCells_GPU(generationInfo *genInfo, Cell_GPU* cells){ 

    int thread_Index = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_Index < genInfo->nCellsThread){
        // int point_count = genInfo->MAX_POINT_PER_CELL;
        // if (genInfo->isTextureUsed) point_count = genInfo->density_images[thread_Index] * (genInfo->MAX_POINT_PER_CELL);
        
        
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
        
        int point_count = currentCell->point_count;
        // currentCell->point_count = point_count;

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
    uint16_t blockSize = BLOCKSIZE;
    int nBlocks = nCellThreads / blockSize + 1 ;
    printf("Number of blocks : %d,\tThreads per block : %d\n", nBlocks, blockSize);
    

    // ALLOCATING CUDA GRID CELLS VARIABLES---------------------------------------------------------------
    
    generationInfo h_genInfo;
    h_genInfo.init_subdiv = INIT_SUBDIV;
    h_genInfo.isTextureUsed = h_isTextureUsed;
    h_genInfo.MAX_POINT_PER_CELL = MAX_POINT_PER_CELL;
    h_genInfo.scale = SCALE;
    h_genInfo.branching = BRANCHING;
    h_genInfo.nCellsThread = nCellThreads;
    
    float *d_density_images; 
    int *d_layer_MileStones;
    
    generationInfo *d_genInfo;
    
    Cell_GPU *d_Cells;
    Cell_GPU * h_cells = new Cell_GPU[nCellThreads];
    
    int numEdges = layer_Milestones[BRANCHING-1]  * MAX_POINT_PER_CELL;
    Edge_GPU *d_edges;
    Edge_GPU *h_edges = new Edge_GPU[numEdges];


    cudaMalloc(&d_layer_MileStones, BRANCHING * sizeof(int));
    cudaMemcpy(d_layer_MileStones, layer_Milestones.data(), BRANCHING*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_density_images, sizeof(float)*weight_maps.size());
    cudaMemcpy(d_density_images, weight_maps.data(), sizeof(float)*weight_maps.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_edges, sizeof(Edge_GPU) * numEdges);

    h_genInfo.density_images = d_density_images;
    h_genInfo.layer_MileStones = d_layer_MileStones;
    h_genInfo.edges = d_edges;
    
    cudaMalloc(&d_genInfo, sizeof(h_genInfo));
    cudaMemcpy(d_genInfo, &h_genInfo, sizeof(h_genInfo), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_Cells, nCellThreads * sizeof(Cell_GPU));
    

    // allocating the arrays inside each cell structure for device memory
    for (int i = 0; i < nCellThreads; i++){
        int numPoints = MAX_POINT_PER_CELL;
        if (h_isTextureUsed) int numPoints = weight_maps[i] * MAX_POINT_PER_CELL;
        
        float *d_points = nullptr;
        point_Info *d_pointsInfo = nullptr;
        
        cudaMalloc((void**)&d_points, numPoints * 3 * sizeof(float));
        cudaMalloc((void**)&d_pointsInfo, numPoints * sizeof(float));
        
        h_cells[i].points = d_points;   //store the pointer to device-points-array inside in host-points-array
        h_cells[i].pointsInfo = d_pointsInfo;
        h_cells[i].point_count = numPoints;
        
    }
    
    // copy h_cells to d_cells with d_cells now contains the pointers to pre allocated inner structures and array
    cudaMemcpy(d_Cells, h_cells, nCellThreads * sizeof(Cell_GPU), cudaMemcpyHostToDevice);

    // COMPUTING ON DEVICE ---------------------------------------------------------------------------------

    // ---------------- POINT GENERATION ---------------- //
    std::cout << "Point Generation..."<< std::endl;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1, 0);

    generateCells_GPU<<<nBlocks, blockSize>>>(d_genInfo, d_Cells);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    float pointGenTime = 0;
    cudaEventElapsedTime(&pointGenTime, start1, stop1);
    
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    std::cout << "CUDA RUN TIME (Point generation): " << pointGenTime << std::endl;


    // ---------------- EDGE CONNECTION ---------------- //
    std::cout << "Edge connection..."<< std::endl;


    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    
    cudaEventRecord(start2);
    edgeConnection<<<nBlocks, blockSize>>>(d_Cells, d_genInfo);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float edgeConnTime = 0;
    cudaEventElapsedTime(&edgeConnTime, start2, stop2);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    std::cout << "CUDA RUN TIME (Edge connection): " << edgeConnTime << std::endl;
    

    // TRANSFERING DATA TO HOST ---------------------------------------------------------------------------
    std::cout << "Transfering data to host...";
    float **d_points_array = new float*[nCellThreads]; // array of pointers to all the arrays of float on device
    for (int i = 0; i < nCellThreads; i++){
        d_points_array[i] = h_cells[i].points; // copied from previously saved device pointer value
    }
    
    auto start3 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nCellThreads; i++){
        int numPoints = MAX_POINT_PER_CELL;
        if (h_isTextureUsed) int numPoints = weight_maps[i] * MAX_POINT_PER_CELL;
        
        float *h_currCellPoints = new float[numPoints * 3];
        point_Info * h_currCellPointsInfo = new point_Info[numPoints];
        
        cudaMemcpy(h_currCellPoints, h_cells[i].points, numPoints * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_currCellPointsInfo, h_cells[i].pointsInfo, numPoints * sizeof(point_Info), cudaMemcpyDeviceToHost);
        
        h_cells[i].points = h_currCellPoints;
        h_cells[i].pointsInfo = h_currCellPointsInfo;

    }
    cudaMemcpy(h_edges, d_edges, sizeof(Edge_GPU) * numEdges, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << " done !"<< std::endl;

    auto stop3 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop3 - start3);
    std::cout << "Data transfert duration : " << duration.count() << " ms" << std::endl;
    
    // CLEANING UP -------------------------------------------------------------------------------------
    cudaFree(d_Cells);
    cudaFree(d_density_images);
    cudaFree(d_layer_MileStones);
    cudaFree(d_genInfo);
    cudaFree(d_edges);

    for (int i = 0; i < nCellThreads; i++){
        cudaFree(d_points_array[i]); // cleaning up previously saved device pointer value
    }
    // ---------------------------------------------------------------------------------------------------

    //std::cout <<"Point list: \n";

    for (int c = 0; c < nCellThreads; c++){

        for (int p = 0; p < h_cells[c].point_count; p++){
            // std::cout << "x: " << h_cells[c].points[p*3] << " " << "y: " << h_cells[c].points[p*3+1] << " " << "z: " << h_cells[c].points[p*3+2] << "\n";
            // std::cout << p << std::endl;
            
        }
    }

    //std::cout << "\nEdges list: \n";
    for (int e = 0; e < numEdges; e++){
        // std::cout << "p1: " << h_edges[e].p1 << " p2: " << h_edges[e].p2 << "\n";
    }

    std::cout << std::endl;

    // std::cout << std::fixed << std::setprecision(6) << h_cells[0].points[0*3] << " " << h_cells[0].points[0*3 + 1] << " " << h_cells[0].points[0*3 + 2] << std::endl;
}


