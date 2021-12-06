cd build
cmake -D VCPKG_TARGET_TRIPLET=x64-windows -D CMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake ..
msbuild RayTracerX.sln /t:Rebuild /p:Configuration=Release
