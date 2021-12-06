## How to build

`git submodule update --init --recursive`

**On Windows:**

1. Install Visual Studio 2022
2. Open *x64 Native Tools Command Prompt for VS 2022*
3. cd to project root
4. `.\configure`
5. `.\build`
6. To run, find executive under the `bin_x64\Release` folder
7. To develop, open `build\RayTracerX.sln`

**On macOS/Linux:**

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`
