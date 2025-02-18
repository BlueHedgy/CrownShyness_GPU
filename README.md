CrownShyness
========================
FIRST OFF:
This code is a continuation of my internship work at MAVERICK, INRIA.
All credit on research topic belongs to *https://github.com/ThibaultTricard*
and the laboratory.

Some few lines about how all these work:

## Cloning the repo
Clone the code by running the following lines:

```console
    git clone https://github.com/BlueHedgy/CrownShyness_GPU.git
    git submodule update --init --recursive
```

## How to compile and run:
This code has been tested on Windows 11 SDK and Ubuntu 22.04, 24.04 LTS family.

Vulkan SDK is required to make the code work (specifically for LavaCake):
- Download instructions are at: https://vulkan.lunarg.com/home/welcome
- While on Ubuntu, you may be prompted a series of missing development packages, just install all of them

How to get it running: 

- The code uses CMake to manage the compilation, install it:
```bash
    sudo apt install cmake
```

or 

```powershell
    winget install cmake -s winget
```

- Using terminal (Powershell recommended if using Windows), navigate to code and run the following commands:
```console
    cd code
    mkdir build && cd build
    cmake ..
    cmake --build .
```

- Using Visual Studio Code (vscode)
    - Install cmake extension
    - simply click build and then run on the bottom left of the screen (extension's functionality)

By default a ***graph.obj*** file will be generated upon running the compiled file inside the **bin** folder in the top directory.

You can open and view the output via 3D softwares such as Blender, Maya, 3DSMax, etc...  
**NOTE:** the code is using **Z up, Y forward** as coordinate system, change the import settings accordingly in your preview software to avoid confusion.

## Repository map
- The code uses a config file as input
    - **config.json** is located in config folder
    - You can setup multiple profiles and load them by name in the console while running
  
- All data structures used is in ***headers/dataStructures.h***
- ***jitterGrid.cpp*** contain the functions that control the grids generation and inital trees' straight edges connection
- ***utils.cpp*** contains the helper functions that deal with input output of the program:
    - **load_Config_Profile()**: self-explanatory
    - **user_density_map()**  : a universal function that turns all image into useable matrix of values
    - **write_to_OBJ()**: self_explanatory

- ***forest_control.cpp*** contains ALL of the functions that manipulate the generated trees:
    - filter_trees(): set branches count of the tree to -1 and prevent it from being written into output if smaller than a set threshold
    - pointFromCoord() : self-explanatory
    - forest_height() : see comment in the code
    - **crownShyness()** : shrink the trees and its foliage clumps toward the center of the tree, or the center of the clumps (multi layers shrinkage)
    - branch_styling() : Kinda deprecated, see comments in the code
    - lerp() & de_Casteljau_Algo() : helper functions for straight edges to curve
    - addSplineToTrees() : self-explanatory
    - **trunkToSpline()**: turn the tree trunks into curves
    - **edgeToSpline()** : transform the straight edges into spline curves (**Branches**), with 2 control points each.


## Stuffs to look into
- Currently the "direction" used in calculation of control points for the spline curves does not make much sense yet:
    - It should probably be more reliant on the edge length, direction, grid layer, etc..
    - It should probably be more random also
    - It should create a flow in the foliage clumps of the trees

- Output curves information, vertex weight into a file format somehow
- Maybe make the trees aware of the angle of the surface it grows on (e.g based on input terrain height texture)

.  
.  
.  
.  
.  
.  
.  
.  

## Side notes:
- json for modern c++ (nlohmann): used for config parsing
- stb (n0thing)                 : used for parsing images
- LavaCake (Thibault Tricard)   : used for vector, matrix calculation
  
Any questions regarding this ***specific*** version of the code send me an email at: *ngphuhung2000@gmail.com*
