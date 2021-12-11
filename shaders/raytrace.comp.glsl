#version 460

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, set = 0, scalar) buffer storageBuffer
{
  vec3 imageData[];
};

layout(binding = 1, set = 0) uniform accelerationStructureEXT tlas;
layout(binding = 2, set = 0, scalar) buffer Vertices
{
  vec3 vertices[];
};
layout(binding = 3, set = 0, scalar) buffer Indices
{
  uint indices[];
};

// Random number generation using pcg32i_random_t, using inc = 1. Our random state is a uint.
uint stepRNG(uint rngState)
{
  return rngState * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
  // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
  rngState  = stepRNG(rngState);
  uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
  word      = (word >> 22) ^ word;
  return float(word) / 4294967295.0f;
}

// Returns the color of the sky in a given direction (in linear color space)
vec3 diffuseSkyColor(vec3 direction)
{
  // sky doesn't have a color in diffuse reflection
  // demo light
  vec3 light = vec3(5.459804, 10.568624, -4.02205);
  // diffuse color might return negative values
  return dot(normalize(direction), normalize(light)) * 0.478 * vec3(1.0, 1.0, 1.0);
}

vec3 screenColor(vec3 c1, vec3 c2) {
  return vec3(1.0, 1.0, 1.0) - (vec3(1.0, 1.0, 1.0) - c1) * (vec3(1.0, 1.0, 1.0) - c2);
}

vec3 specularSkyColor(vec3 direction)
{
  vec3 color = vec3(0.235294, 0.67451, 0.843137);
  // demo light
  vec3 light = vec3(5.459804, 10.568624, -4.02205);
  if (dot(normalize(direction), normalize(light)) > 0) {
    return screenColor(color, dot(normalize(direction), normalize(light)) * vec3(1.0, 1.0, 1.0));
  }
  return color;
}

struct HitInfo
{
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
};

HitInfo getObjectHitInfo(rayQueryEXT rayQuery)
{
  HitInfo result;
  // Get the ID of the triangle
  const int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);

  // Get the indices of the vertices of the triangle
  const uint i0 = indices[3 * primitiveID + 0];
  const uint i1 = indices[3 * primitiveID + 1];
  const uint i2 = indices[3 * primitiveID + 2];

  // Get the vertices of the triangle
  const vec3 v0 = vertices[i0];
  const vec3 v1 = vertices[i1];
  const vec3 v2 = vertices[i2];

  // Get the barycentric coordinates of the intersection
  vec3 barycentrics = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(rayQuery, true));
  barycentrics.x    = 1.0 - barycentrics.y - barycentrics.z;

  // Compute the coordinates of the intersection
  const vec3 objectPos = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
  // For the main tutorial, object space is the same as world space:
  result.worldPosition = objectPos;

  // Compute the normal of the triangle in object space, using the right-hand rule:
  //    v2      .
  //    |\      .
  //    | \     .
  //    |/ \    .
  //    /   \   .
  //   /|    \  .
  //  L v0---v1 .
  // n
  const vec3 objectNormal = normalize(cross(v1 - v0, v2 - v0));
  // For the main tutorial, object space is the same as world space:
  result.worldNormal = objectNormal;

  result.color = vec3(0.7f);

  return result;
}

void main()
{
  // The resolution of the buffer, which in this case is a hardcoded vector
  // of 2 unsigned integers:
  const uvec2 resolution = uvec2(800, 600);

  // Get the coordinates of the pixel for this invocation:
  //
  // .-------.-> x
  // |       |
  // |       |
  // '-------'
  // v
  // y
  const uvec2 pixel = gl_GlobalInvocationID.xy;

  // If the pixel is outside of the image, don't do anything:
  if((pixel.x >= resolution.x) || (pixel.y >= resolution.y))
  {
    return;
  }

  // State of the random number generator.
  uint rngState = resolution.x * pixel.y + pixel.x;  // Initial seed

  const mat4 cameraToWorld = mat4(-0.972776, 0, -0.231748, 0, -0.114956, 0.8683, 0.482536, 0, 0.201227, 0.49604, -0.844661, 0, 6.696465, 22.721296, -30.097976, 1);
  const float fov = 36.87;
  float scale = tan(radians(fov * 0.5));
  float imageAspectRatio = resolution.x / float(resolution.y);

  // The sum of the colors of all of the samples.
  vec3 summedPixelColor = vec3(0.0);

  // Limit the kernel to trace at most 64 samples.
  const int NUM_SAMPLES = 64;
  for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
  {
    const vec2 randomPixelCenter = vec2(pixel) + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    float x = (2 * (float(randomPixelCenter.x) + 0.5) / float(resolution.x) - 1) * imageAspectRatio * scale;
    float y = (1 - 2 * (float(randomPixelCenter.y) + 0.5) / float(resolution.y)) * scale;
    vec3 rayDirection = vec3(cameraToWorld * vec4(x, y, -1.0, 0.0));
    rayDirection      = normalize(rayDirection);
    vec3 rayOrigin    = vec3(cameraToWorld * vec4(0.0, 0.0, 0.0, 1.0));

    vec3 accumulatedRayColor = vec3(1.0);  // The amount of light that made it to the end of the current ray.

    bool isDiffuse = false;
    // Limit the kernel to trace at most 32 segments.
    for(int tracedSegments = 0; tracedSegments < 32; tracedSegments++)
    {
      // Trace the ray and see if and where it intersects the scene!
      // First, initialize a ray query object:
      rayQueryEXT rayQuery;
      rayQueryInitializeEXT(rayQuery,              // Ray query
                            tlas,                  // Top-level acceleration structure
                            gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            rayOrigin,             // Ray origin
                            0.0,                   // Minimum t-value
                            rayDirection,          // Ray direction
                            10000.0);              // Maximum t-value

      // Start traversal, and loop over all ray-scene intersections. When this finishes,
      // rayQuery stores a "committed" intersection, the closest intersection (if any).
      while(rayQueryProceedEXT(rayQuery))
      {
      }

      // Get the type of committed (true) intersection - nothing, a triangle, or
      // a generated object
      if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
      {
        // Ray hit a triangle
        HitInfo hitInfo = getObjectHitInfo(rayQuery);

        // Apply color absorption
        accumulatedRayColor *= hitInfo.color;

        // Flip the normal so it points against the ray direction:
        hitInfo.worldNormal = faceforward(hitInfo.worldNormal, rayDirection, hitInfo.worldNormal);

        // Start a new ray at the hit position, but offset it slightly along the normal:
        rayOrigin = hitInfo.worldPosition + 0.0001 * hitInfo.worldNormal;
        if (!isDiffuse) {
            // Reflect the direction of the ray using the triangle normal:
            rayDirection = reflect(rayDirection, hitInfo.worldNormal);
        } else {
            const float theta = 6.2831853 * stepAndOutputRNGFloat(rngState);   // Random in [0, 2pi]
            const float u     = 2.0 * stepAndOutputRNGFloat(rngState) - 1.0;  // Random in [-1, 1]
            const float r     = sqrt(1.0 - u * u);
            rayDirection      = hitInfo.worldNormal + vec3(r * cos(theta), r * sin(theta), u);
            // Then normalize the ray direction:
            rayDirection = normalize(rayDirection);
        }
      }
      else
      {
        // Ray hit the sky
        if (isDiffuse)
            accumulatedRayColor *= diffuseSkyColor(rayDirection); 
        else
            accumulatedRayColor *= specularSkyColor(rayDirection); 
        
        // Sum this with the pixel's other samples.
        // (Note that we treat a ray that didn't find a light source as if it had
        // an accumulated color of (0, 0, 0)).
        summedPixelColor += accumulatedRayColor;
    
        break;
      }
    }
  }

  // Get the index of this invocation in the buffer:
  uint linearIndex       = resolution.x * pixel.y + pixel.x;
  imageData[linearIndex] = summedPixelColor / float(NUM_SAMPLES);  // Take the average
//  const vec3 pixelColor = vec3(float(pixel.x) / resolution.x,  // Red
//                               float(pixel.y) / resolution.y,  // Green
//                               0.0);                           // Blue
//  imageData[linearIndex] = pixelColor;
}