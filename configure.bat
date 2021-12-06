mkdir build
cd build
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
call bootstrap-vcpkg.bat

vcpkg.exe install ^
    glfw3:x64-windows ^
    glew:x64-windows ^
    glm:x64-windows ^
    stb:x64-windows ^
    tinyobjloader:x64-windows

cd ..
cmake -D VCPKG_TARGET_TRIPLET=x64-windows -D CMAKE_TOOLCHAIN_FILE=./vcpkg/scripts/buildsystems/vcpkg.cmake ..
cd ..
