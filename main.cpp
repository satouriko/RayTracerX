// Based on templates from learnopengl.com
#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <array>
#include <filesystem>
#include <glm/glm.hpp>
#include <iostream>
#include <nvh/fileoperations.hpp> // For nvh::loadFile
#include <nvvk/context_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>    // For nvvk::DescriptorSetContainer
#include <nvvk/error_vk.hpp>             // For NVVK_CHECK
#include <nvvk/raytraceKHR_vk.hpp>       // For nvvk::RaytracingBuilderKHR
#include <nvvk/resourceallocator_vk.hpp> // For NVVK memory allocators
#include <nvvk/shaders_vk.hpp>           // For nvvk::createShaderModule
#include <nvvk/structs_vk.hpp>           // For nvvk::make

// settings

static const uint32_t workgroup_width = 16;
static const uint32_t workgroup_height = 8;

static const uint64_t render_width = 800;
static const uint64_t render_height = 600;

const unsigned int SCR_WIDTH = render_width;
const unsigned int SCR_HEIGHT = render_height;

enum MaterialType { kDiffuse, kReflection, kReflectionAndRefraction };

VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device,
                                                     VkCommandPool cmdPool) {
  VkCommandBufferAllocateInfo cmdAllocInfo =
      nvvk::make<VkCommandBufferAllocateInfo>();
  cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandPool = cmdPool;
  cmdAllocInfo.commandBufferCount = 1;
  VkCommandBuffer cmdBuffer;
  NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));
  VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
  return cmdBuffer;
}

void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue,
                                       VkCommandPool cmdPool,
                                       VkCommandBuffer &cmdBuffer) {
  NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
  VkSubmitInfo submitInfo = nvvk::make<VkSubmitInfo>();
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuffer;
  NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
  NVVK_CHECK(vkQueueWaitIdle(queue));
  vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer) {
  VkBufferDeviceAddressInfo addressInfo =
      nvvk::make<VkBufferDeviceAddressInfo>();
  addressInfo.buffer = buffer;
  return vkGetBufferDeviceAddress(device, &addressInfo);
}

void loadPolyMeshFromFile(const char *file,
                          std::vector<std::string> directories,
                          std::vector<uint32_t> &objIndices,
                          std::vector<float> &objVertices,
                          std::vector<float> &objNormals,
                          std::vector<float> &objStCords,
                          std::vector<float> &objIor,
                          float ior) {
  for (auto &directory : directories) {
    auto offset = objVertices.size() / 3;
    std::ifstream ifs;
    try {
      ifs.open(directory + "/" + file);
      if (ifs.fail()) {
        ifs.close();
        continue;
      }
      std::stringstream ss;
      ss << ifs.rdbuf();
      uint32_t numFaces;
      ss >> numFaces;
      std::unique_ptr<uint32_t[]> faceIndex(new uint32_t[numFaces]);
      uint32_t vertsIndexArraySize = 0;
      // reading face index array
      for (uint32_t i = 0; i < numFaces; ++i) {
        ss >> faceIndex[i];
        vertsIndexArraySize += faceIndex[i];
      }
      std::unique_ptr<uint32_t[]> vertsIndex(new uint32_t[vertsIndexArraySize]);
      uint32_t vertsArraySize = 0;
      // reading vertex index array
      for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
        ss >> vertsIndex[i];
        if (vertsIndex[i] > vertsArraySize)
          vertsArraySize = vertsIndex[i];
      }
      vertsArraySize += 1;
      // reading vertices
      std::unique_ptr<glm::vec3[]> verts(new glm::vec3[vertsArraySize]);
      for (uint32_t i = 0; i < vertsArraySize; ++i) {
        ss >> verts[i].x >> verts[i].y >> verts[i].z;
        objVertices.push_back(verts[i].x);
        objVertices.push_back(verts[i].y);
        objVertices.push_back(verts[i].z);
      }
      // reading normals
      std::unique_ptr<glm::vec3[]> normals(new glm::vec3[vertsIndexArraySize]);
      for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
        ss >> normals[i].x >> normals[i].y >> normals[i].z;
        objNormals.push_back(normals[i].x);
        objNormals.push_back(normals[i].y);
        objNormals.push_back(normals[i].z);
      }
      // reading st coordinates
      std::unique_ptr<glm::vec3[]> st(new glm::vec3[vertsIndexArraySize]);
      for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
        ss >> st[i].x >> st[i].y;
        objStCords.push_back(st[i].x);
        objStCords.push_back(st[i].y);
      }
      uint32_t k = 0, maxVertIndex = 0, numTris = 0;
      // find out how many triangles we need to create for this mesh
      for (uint32_t i = 0; i < numFaces; ++i) {
        numTris += faceIndex[i] - 2;
        for (uint32_t j = 0; j < faceIndex[i]; ++j)
          if (vertsIndex[k + j] > maxVertIndex)
            maxVertIndex = vertsIndex[k + j];
        k += faceIndex[i];
      }
      maxVertIndex += 1;
      // generate the triangle index array and set normals and st coordinates
      for (uint32_t i = 0, k = 0; i < numFaces; ++i) { // for each  face
        for (uint32_t j = 0; j < faceIndex[i] - 2;
             ++j) { // for each triangle in the face
          objIndices.push_back(offset + vertsIndex[k]);
          objIndices.push_back(offset + vertsIndex[k + j + 1]);
          objIndices.push_back(offset + vertsIndex[k + j + 2]);
          objIor.push_back(ior);
        }
        k += faceIndex[i];
      }
      ifs.close();
      return;
    } catch (...) {
      ifs.close();
    }
  }
}

void render(int &width, int &height, unsigned char *&data) {
  // Create the Vulkan context, consisting of an instance, device, physical
  // device, and queues.
  nvvk::ContextCreateInfo
      deviceInfo; // One can modify this to load different extensions or pick
                  // the Vulkan core version
  deviceInfo.apiMajor = 1; // Specify the version of Vulkan we'll use
  deviceInfo.apiMinor = 2;
  deviceInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures =
      nvvk::make<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
  deviceInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                false, &asFeatures);
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures =
      nvvk::make<VkPhysicalDeviceRayQueryFeaturesKHR>();
  deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false,
                                &rayQueryFeatures);

  nvvk::Context context;    // Encapsulates device state in a single object
  context.init(deviceInfo); // Initialize the context
  assert(asFeatures.accelerationStructure == VK_TRUE &&
         rayQueryFeatures.rayQuery == VK_TRUE);
  nvvk::ResourceAllocatorDedicated allocator;
  allocator.init(context, context.m_physicalDevice);

  // Create a buffer
  VkDeviceSize bufferSizeBytes =
      render_width * render_height * 3 * sizeof(float);
  VkBufferCreateInfo bufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
  bufferCreateInfo.size = bufferSizeBytes;
  bufferCreateInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that the CPU can read this
  // buffer's memory. VK_MEMORY_PROPERTY_HOST_CACHED_BIT means that the CPU
  // caches this memory. VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means that the CPU
  // side of cache management is handled automatically, with potentially slower
  // reads/writes.
  nvvk::Buffer buffer =
      allocator.createBuffer(bufferCreateInfo,                        //
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT      //
                                 | VK_MEMORY_PROPERTY_HOST_CACHED_BIT //
                                 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  std::vector<uint32_t> objIndices;
  std::vector<float> objVertices;
  std::vector<float> objNormals;
  std::vector<float> objStCords;
  std::vector<float> objIor;

  auto exePath = std::filesystem::current_path().generic_string();

  loadPolyMeshFromFile("geometry/backdrop.geo", {exePath, exePath + "/.."},
                       objIndices, objVertices, objNormals, objStCords, objIor, 0.0);
  loadPolyMeshFromFile("geometry/cylinder.geo", {exePath, exePath + "/.."},
                       objIndices, objVertices, objNormals, objStCords, objIor, 1.5);
  loadPolyMeshFromFile("geometry/pen.geo", {exePath, exePath + "/.."},
                       objIndices, objVertices, objNormals, objStCords, objIor, 0.0);

  // Shader loading and pipeline creation
  VkShaderModule rayTraceModule = nvvk::createShaderModule(
      context, nvh::loadFile("shaders/raytrace.comp.glsl.spv", true,
                             {exePath, exePath + "/.."}));

  // Describes the entrypoint and the stage to use for this shader module in the
  // pipeline
  VkPipelineShaderStageCreateInfo shaderStageCreateInfo =
      nvvk::make<VkPipelineShaderStageCreateInfo>();
  shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageCreateInfo.module = rayTraceModule;
  shaderStageCreateInfo.pName = "main";

  // Create the command pool
  VkCommandPoolCreateInfo cmdPoolInfo = nvvk::make<VkCommandPoolCreateInfo>();
  cmdPoolInfo.queueFamilyIndex = context.m_queueGCT;
  VkCommandPool cmdPool;
  NVVK_CHECK(vkCreateCommandPool(context, &cmdPoolInfo, nullptr, &cmdPool));

  // Upload the vertex and index buffers to the GPU.
  nvvk::Buffer vertexBuffer, indexBuffer, iorBuffer;
  {
    // Start a command buffer for uploading the buffers
    VkCommandBuffer uploadCmdBuffer =
        AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
    // We get these buffers' device addresses, and use them as storage buffers
    // and build inputs.
    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    vertexBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);
    indexBuffer = allocator.createBuffer(uploadCmdBuffer, objIndices, usage);
    iorBuffer = allocator.createBuffer(uploadCmdBuffer, objIor, usage);
    EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool,
                                      uploadCmdBuffer);
    allocator.finalizeAndReleaseStaging();
  }
  // Describe the bottom-level acceleration structure (BLAS)
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;
  {
    nvvk::RaytracingBuilderKHR::BlasInput blas;
    // Get the device addresses of the vertex and index buffers
    VkDeviceAddress vertexBufferAddress =
        GetBufferDeviceAddress(context, vertexBuffer.buffer);
    VkDeviceAddress indexBufferAddress =
        GetBufferDeviceAddress(context, indexBuffer.buffer);
    // Specify where the builder can find the vertices and indices for
    // triangles, and their formats:
    VkAccelerationStructureGeometryTrianglesDataKHR triangles =
        nvvk::make<VkAccelerationStructureGeometryTrianglesDataKHR>();
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexBufferAddress;
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = static_cast<uint32_t>(objVertices.size() / 3 - 1);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBufferAddress;
    triangles.transformData.deviceAddress = 0; // No transform
    // Create a VkAccelerationStructureGeometryKHR object that says it handles
    // opaque triangles and points to the above:
    VkAccelerationStructureGeometryKHR geometry =
        nvvk::make<VkAccelerationStructureGeometryKHR>();
    geometry.geometry.triangles = triangles;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    blas.asGeometry.push_back(geometry);
    // Create offset info that allows us to say how many triangles and vertices
    // to read
    VkAccelerationStructureBuildRangeInfoKHR offsetInfo;
    offsetInfo.firstVertex = 0;
    offsetInfo.primitiveCount =
        static_cast<uint32_t>(objIndices.size() / 3); // Number of triangles
    offsetInfo.primitiveOffset = 0;
    offsetInfo.transformOffset = 0;
    blas.asBuildOffsetInfo.push_back(offsetInfo);
    blases.push_back(blas);
  }
  // Create the BLAS
  nvvk::RaytracingBuilderKHR raytracingBuilder;
  raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
  raytracingBuilder.buildBlas(
      blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  std::vector<VkAccelerationStructureInstanceKHR> instances;
  {
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = nvvk::toTransformMatrixKHR(
        nvmath::mat4f(1)); // Set the instance transform to the identity matrix
    instance.instanceCustomIndex =
        0; // 24 bits accessible to ray shaders via
           // rayQueryGetIntersectionInstanceCustomIndexEXT
    instance.accelerationStructureReference =
        raytracingBuilder.getBlasDeviceAddress(
            0); // The address of the BLAS in `blases` that this instance points
                // to
    // Used for a shader offset index, accessible via
    // rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags =
        VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR; // How to
                                                                   // trace this
                                                                   // instance
    instance.mask = 0xFF;
    instances.push_back(instance);
  }
  raytracingBuilder.buildTlas(
      instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

  // Here's the list of bindings for the descriptor set layout, from
  // raytrace.comp.glsl: 0 - a storage buffer (the buffer `buffer`) 1 - an
  // acceleration structure (the TLAS) That's it for now!
  nvvk::DescriptorSetContainer descriptorSetContainer(context);
  descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(
      1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
      VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
  descriptorSetContainer.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                    VK_SHADER_STAGE_COMPUTE_BIT);
  // Create a layout from the list of bindings
  descriptorSetContainer.initLayout();
  // Create a descriptor pool from the list of bindings with space for 1 set,
  // and allocate that set
  descriptorSetContainer.initPool(1);
  // Create a simple pipeline layout from the descriptor set layout:
  descriptorSetContainer.initPipeLayout();
  // Write values into the descriptor set.
  std::array<VkWriteDescriptorSet, 5> writeDescriptorSets;
  // 0
  VkDescriptorBufferInfo descriptorBufferInfo{};
  descriptorBufferInfo.buffer = buffer.buffer; // The VkBuffer object
  descriptorBufferInfo.range =
      bufferSizeBytes; // The length of memory to bind; offset is 0.
  writeDescriptorSets[0] = descriptorSetContainer.makeWrite(
      0 /*set index*/, 0 /*binding*/, &descriptorBufferInfo);
  // 1
  VkWriteDescriptorSetAccelerationStructureKHR descriptorAS =
      nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
  VkAccelerationStructureKHR tlasCopy =
      raytracingBuilder
          .getAccelerationStructure(); // So that we can take its address
  descriptorAS.accelerationStructureCount = 1;
  descriptorAS.pAccelerationStructures = &tlasCopy;
  writeDescriptorSets[1] =
      descriptorSetContainer.makeWrite(0, 1, &descriptorAS);
  // 2
  VkDescriptorBufferInfo vertexDescriptorBufferInfo{};
  vertexDescriptorBufferInfo.buffer = vertexBuffer.buffer;
  vertexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
  writeDescriptorSets[2] =
      descriptorSetContainer.makeWrite(0, 2, &vertexDescriptorBufferInfo);
  // 3
  VkDescriptorBufferInfo indexDescriptorBufferInfo{};
  indexDescriptorBufferInfo.buffer = indexBuffer.buffer;
  indexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
  writeDescriptorSets[3] =
      descriptorSetContainer.makeWrite(0, 3, &indexDescriptorBufferInfo);
  // 4
  VkDescriptorBufferInfo iorDescriptorBufferInfo{};
  iorDescriptorBufferInfo.buffer = iorBuffer.buffer;
  iorDescriptorBufferInfo.range = VK_WHOLE_SIZE;
  writeDescriptorSets[4] =
      descriptorSetContainer.makeWrite(0, 4, &iorDescriptorBufferInfo);
  vkUpdateDescriptorSets(
      context, // The context
      static_cast<uint32_t>(
          writeDescriptorSets.size()), // Number of VkWriteDescriptorSet objects
      writeDescriptorSets.data(), // Pointer to VkWriteDescriptorSet objects
      0, nullptr); // An array of VkCopyDescriptorSet objects (unused)

  // Create the compute pipeline
  VkComputePipelineCreateInfo pipelineCreateInfo =
      nvvk::make<VkComputePipelineCreateInfo>();
  pipelineCreateInfo.stage = shaderStageCreateInfo;
  pipelineCreateInfo.layout = descriptorSetContainer.getPipeLayout();
  ;
  // Don't modify flags, basePipelineHandle, or basePipelineIndex
  VkPipeline computePipeline;
  NVVK_CHECK(vkCreateComputePipelines(
      context,                // Device
      VK_NULL_HANDLE,         // Pipeline cache (uses default)
      1, &pipelineCreateInfo, // Compute pipeline create info
      VK_NULL_HANDLE,         // Allocator (uses default)
      &computePipeline));     // Output

  // Create and start recording a command buffer
  VkCommandBuffer cmdBuffer =
      AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);

  // Bind the descriptor set
  VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
  vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          descriptorSetContainer.getPipeLayout(), 0, 1,
                          &descriptorSet, 0, nullptr);
  // Bind the compute shader pipeline
  vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
  // Run the compute shader with one workgroup for now
  // Run the compute shader with enough workgroups to cover the entire buffer:
  vkCmdDispatch(
      cmdBuffer,
      (uint32_t(render_width) + workgroup_width - 1) / workgroup_width,
      (uint32_t(render_height) + workgroup_height - 1) / workgroup_height, 1);

  // Add a command that says "Make it so that memory writes by the
  // vkCmdFillBuffer call are available to read from the CPU." (In other words,
  // "Flush the GPU caches so the CPU can read the data.") To do this, we use a
  // memory barrier.
  VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
  memoryBarrier.srcAccessMask =
      VK_ACCESS_SHADER_WRITE_BIT;                        // Make shader writes
  memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT; // Readable by the CPU
  vkCmdPipelineBarrier(
      cmdBuffer,                            // The command buffer
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // From the transfer stage
      VK_PIPELINE_STAGE_HOST_BIT,           // To the CPU
      0,                                    // No special flags
      1, &memoryBarrier,                    // An array of memory barriers
      0, nullptr, 0, nullptr);              // No other barriers
  // End and submit the command buffer, then wait for it to finish:
  EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool,
                                    cmdBuffer);
  void *data_v = allocator.map(buffer);
  float *data_f = reinterpret_cast<float *>(data_v);
  width = render_width, height = render_height; // keep it in powers of 2!
  unsigned char *image =
      (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int idx = (i * width + j) * 3;
      int idgl = ((height - i - 1) * width + j) * 3;
      image[idgl] = (unsigned char)(255 * data_f[idx]);
      image[idgl + 1] = (unsigned char)(255 * data_f[idx + 1]);
      image[idgl + 2] = (unsigned char)(255 * data_f[idx + 2]);
    }
  }
  data = image;
  allocator.unmap(buffer);

  allocator.destroy(buffer);
  vkDestroyPipeline(context, computePipeline, nullptr);
  vkDestroyShaderModule(context, rayTraceModule, nullptr);
  descriptorSetContainer.deinit();
  raytracingBuilder.destroy();
  allocator.destroy(vertexBuffer);
  allocator.destroy(indexBuffer);
  vkDestroyCommandPool(context, cmdPool, nullptr);
  allocator.deinit();
  context.deinit(); // Don't forget to clean up at the end of the program!
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

const char *vertexShaderSource = "#version 330 core\n"
                                 "layout (location = 0) in vec3 aPos;\n"
                                 "layout (location = 1) in vec3 aColor;\n"
                                 "layout (location = 2) in vec2 aTexCoord;\n"
                                 "out vec3 ourColor;\n"
                                 "out vec2 TexCoord;\n"
                                 "void main()\n"
                                 "{\n"
                                 "gl_Position = vec4(aPos, 1.0);\n"
                                 "ourColor = aColor;\n"
                                 "TexCoord = vec2(aTexCoord.x, aTexCoord.y);\n"
                                 "}\0";

const char *fragmentShaderSource =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec3 ourColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D texture1;\n"
    "void main()\n"
    "{\n"
    "   FragColor = texture(texture1, TexCoord);\n"
    "}\n\0";

int main() {

  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // glfw window creation
  // --------------------
  GLFWwindow *window =
      glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Display RGB Array", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // // GLEW: load all OpenGL function pointers
  glewInit();

  // build and compile the shaders
  // ------------------------------------
  // vertex shader
  unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  // check for shader compile errors
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
              << infoLog << std::endl;
  }
  // fragment shader
  unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  // check for shader compile errors
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
              << infoLog << std::endl;
  }
  // link shaders
  unsigned int shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  // check for linking errors
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
              << infoLog << std::endl;
  }
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  float vertices[] = {
      // positions          // colors           // texture coords
      0.5f,  0.5f,  0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top right
      0.5f,  -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom right
      -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
      -0.5f, 0.5f,  0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f  // top left
  };
  unsigned int indices[] = {
      0, 1, 3, // first triangle
      1, 2, 3  // second triangle
  };
  unsigned int VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                        (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // texture coord attribute
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                        (void *)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  // load and create a texture
  // -------------------------
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D,
                texture); // all upcoming GL_TEXTURE_2D operations now have
  // effect on this texture object
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                  GL_REPEAT); // set texture wrapping to GL_REPEAT (default
  // wrapping method)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_NEAREST_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  int width = 0, height = 0;
  unsigned char *data = nullptr;
  render(width, height, data);

  if (data) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
  } else {
    std::cout << "Failed to load texture" << std::endl;
  }

  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {
    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // bind Texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // render container
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse
    // moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina
  // displays.
  glViewport(0, 0, width, height);
}