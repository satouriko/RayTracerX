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
layout(binding = 4, set = 0, scalar) buffer Iors
{
  float iors[];
};
layout(binding = 5, set = 0, scalar) buffer Normals
{
  vec3 normals[];
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

struct HitInfo
{
  vec3 color;
  vec3 worldPosition;
  vec3 worldNormal;
  float ior;
};

float getObjectIor(rayQueryEXT rayQuery)
{
  const int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
  return iors[2 * primitiveID];
}

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
  if (iors[2 * primitiveID + 1] == 1.0) {
      const vec3 n0 = normalize(normals[3 * primitiveID]);
      const vec3 n1 = normalize(normals[3 * primitiveID + 1]);
      const vec3 n2 = normalize(normals[3 * primitiveID + 2]);
      const vec3 objectNormal = n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z;
      // For the main tutorial, object space is the same as world space:
      result.worldNormal = normalize(objectNormal);
  } else {
      result.worldNormal = normalize(cross(v1 - v0, v2 - v0));
  }

  result.color = vec3(0.7f);

  result.ior = iors[2 * primitiveID];

  return result;
}

vec3 lightDir = -normalize(vec3(5.459804, 10.568624, -4.02205));

float fresnel(vec3 I, vec3 N, float ior)
{
    float kr = 0.0;
    float cosi = clamp(dot(I, N), -1, 1);
    float etai = 1, etat = ior;
    if (cosi > 0) { 
      etai = etat;
      etat = 1;
    }
    // Compute sini using Snell's law
    float sint = etai / etat * sqrt(max(0.f, 1 - cosi * cosi));
    // Total internal reflection
    if (sint >= 1) {
        kr = 1;
    }
    else {
        float cost = sqrt(max(0.f, 1 - sint * sint));
        cosi = abs(cosi);
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
        kr = (Rs * Rs + Rp * Rp) / 2;
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;
    return kr;
}

vec3 refract1(vec3 I, vec3 N, float ior)
{
    float cosi = clamp(dot(I,N), -1, 1);
    float etai = 1, etat = ior;
    vec3 n = N;
    if (cosi < 0) { cosi = -cosi; }
    else {
      etai = ior;
      etat = 1;
      n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? vec3(0.0) : eta * I + (eta * cosi - sqrt(k)) * n;
}

bool trace(vec3 rayOrigin, vec3 rayDirection) {
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery,              // Ray query
                            tlas,                  // Top-level acceleration structure
                            gl_RayFlagsNoOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            rayOrigin,             // Ray origin
                            0.0,                   // Minimum t-value
                            rayDirection,          // Ray direction
                            10000.0);              // Maximum t-value

    // Start traversal, and loop over all ray-scene intersections. When this finishes,
    // rayQuery stores a "committed" intersection, the closest intersection (if any).
    while(rayQueryProceedEXT(rayQuery))
    {
      if (rayQueryGetIntersectionTypeEXT(rayQuery, false) ==
        gl_RayQueryCandidateIntersectionTriangleEXT)
        {
            float ior = getObjectIor(rayQuery);
            if (ior == 0.0) {
              rayQueryConfirmIntersectionEXT(rayQuery);
            }
        }
    }
  return rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT;
}

struct CastRayObj {
  vec3 rayOrigin;
  vec3 rayDirection;
  float coefficient;
  int depth;
};

vec3 castRay(vec3 rayOrigin0, vec3 rayDirection0) {
  const int maxDepth = 32;
  vec3 backgroundColor = vec3(0.235294, 0.67451, 0.843137);
  vec3 hitColor = vec3(0.0);
  CastRayObj stack[maxDepth];
  CastRayObj initial;
  initial.rayOrigin = rayOrigin0;
  initial.rayDirection = rayDirection0;
  initial.coefficient = 1;
  initial.depth = 0;
  int ptr = 0;
  stack[ptr] = initial;
  while (ptr >= 0) {
      CastRayObj cobj = stack[ptr];
      ptr--;
      rayQueryEXT rayQuery;
      rayQueryInitializeEXT(rayQuery,              // Ray query
                                tlas,                  // Top-level acceleration structure
                                gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                                0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                                cobj.rayOrigin,             // Ray origin
                                0.0,                   // Minimum t-value
                                cobj.rayDirection,          // Ray direction
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
          // diffuse
          if (hitInfo.ior == 0.0) {
            // trace by the direction of light to see if shadowed
            hitInfo.worldNormal = faceforward(hitInfo.worldNormal, cobj.rayDirection, hitInfo.worldNormal);

            // Start a new ray at the hit position, but offset it slightly along the normal:
            vec3 hitPoint = hitInfo.worldPosition + 0.0001 * hitInfo.worldNormal;
            bool vis = !trace(hitPoint, -lightDir);
            if (vis)
                hitColor += cobj.coefficient * vec3(0.4) * max(0.0, dot(hitInfo.worldNormal, -lightDir));
          } else {
            vec3 refractionColor = vec3(0.0);
            vec3 reflectionColor = vec3(0.0);
            float kr = fresnel(cobj.rayDirection, hitInfo.worldNormal, hitInfo.ior);
            // compute refraction if it is not a case of total internal reflection
            if (kr < 1) {
               vec3 refractionDirection = normalize(refract1(cobj.rayDirection, hitInfo.worldNormal, hitInfo.ior));
               hitInfo.worldNormal = faceforward(hitInfo.worldNormal,cobj.rayDirection, hitInfo.worldNormal);
               vec3 refractionRayOrig = hitInfo.worldPosition - 0.0001 * hitInfo.worldNormal;
               ptr++;
               CastRayObj nObj;
               nObj.rayOrigin = refractionRayOrig;
               nObj.rayDirection = refractionDirection;
               nObj.coefficient = cobj.coefficient * (1 - kr);
               nObj.depth = cobj.depth + 1;
               if (ptr >= maxDepth || nObj.depth >= maxDepth) {
                 hitColor += nObj.coefficient * backgroundColor;
                 ptr--;
               } else {
                 stack[ptr] = nObj;
               }
            }
            hitInfo.worldNormal = faceforward(hitInfo.worldNormal,cobj.rayDirection, hitInfo.worldNormal);
            vec3 reflectionDirection = normalize(reflect(cobj.rayDirection, hitInfo.worldNormal));
            vec3 reflectionRayOrig = hitInfo.worldPosition + 0.0001 * hitInfo.worldNormal;
            ptr++;
            CastRayObj nObj;
            nObj.rayOrigin = reflectionRayOrig;
            nObj.rayDirection = reflectionDirection;
            nObj.coefficient = cobj.coefficient * kr;
            nObj.depth = cobj.depth + 1;
            if (ptr >= maxDepth || nObj.depth >= maxDepth) {
              hitColor += nObj.coefficient * backgroundColor;
              ptr--;
            } else {
              stack[ptr] = nObj;
            }
          }
        }
        else {
          hitColor += cobj.coefficient * backgroundColor;
        }
    }
  return hitColor;
}

void main()
{
  // The resolution of the buffer, which in this case is a hardcoded vector
  // of 2 unsigned integers:
  const uvec2 resolution = uvec2(1024, 747);

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
  const int NUM_SAMPLES = 1;
  for(int sampleIdx = 0; sampleIdx < NUM_SAMPLES; sampleIdx++)
  {
    const vec2 randomPixelCenter = vec2(pixel); // + vec2(stepAndOutputRNGFloat(rngState), stepAndOutputRNGFloat(rngState));
    float x = (2 * (float(randomPixelCenter.x) + 0.5) / float(resolution.x) - 1) * imageAspectRatio * scale;
    float y = (1 - 2 * (float(randomPixelCenter.y) + 0.5) / float(resolution.y)) * scale;
    vec3 rayDirection = vec3(cameraToWorld * vec4(x, y, -1.0, 0.0));
    rayDirection      = normalize(rayDirection);
    vec3 rayOrigin    = vec3(cameraToWorld * vec4(0.0, 0.0, 0.0, 1.0));

    summedPixelColor += castRay(rayOrigin, rayDirection);
  }

  // Get the index of this invocation in the buffer:
  uint linearIndex       = resolution.x * pixel.y + pixel.x;
  imageData[linearIndex] = summedPixelColor / float(NUM_SAMPLES);  // Take the average
//  const vec3 pixelColor = vec3(float(pixel.x) / resolution.x,  // Red
//                               float(pixel.y) / resolution.y,  // Green
//                               0.0);                           // Blue
//  imageData[linearIndex] = pixelColor;
}