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
cd ..
