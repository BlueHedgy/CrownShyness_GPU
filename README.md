CrownShyness
========================

Some few lines about how all these work:

## Cloning the repo
Clone the code by running the following lines:

```console
    git clone git@gitlab.inria.fr:ttricard/crownshyness.git
    git update submodules --init --recursive
```

## How to compile and run:
- Using terminal, navigate to code and run the following commands:
```console
    cd code
    mkdir build && cd build
    cmake --build .
```

- Using Visual Studio Code (vscode)
    - Install cmake extension
    - simply click build and then run on the bottom left of the screen (extension's functionality)

By default a ***graph.obj*** file will be generated upon running the compiled file inside the **bin** folder in the top directory.

You can open and view the output via 3D softwares such as Blender, Maya, 3DSMax, etc...  
NOTE: the code is using Z up, Y forward as coordinate system, change the import settings accordingly in your preview software to avoid confusion.

## The code uses a config file as input
- config.json is located in config folder
- setup multiple profiles and load them by name in the console while running

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
  
Any questions regarding this ***specific*** version of the code send me an email at: *ngphuhung2000@gmail.com*