#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <istream>
#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#define PI               3.14159265358979323f
#define TWO_PI           6.28318530717958648f
#define FOUR_PI          12.5663706143591729f
#define INV_PI           0.31830988618379067f
#define INV_TWO_PI       0.15915494309f
#define INV_FOUR_PI      0.07957747154594767f
#define PI_OVER_TWO      1.57079632679489662f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON 0.00001f

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line);

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
    bool SortByMaterial;
    bool UseBVH;
    bool ACESFilm;
    bool NoGammaCorrection;
    float focalLength;
    float apertureSize;
    float theta, phi;
    glm::vec3 cameraLookAt;
    float zoom;
};

namespace utilityCore
{
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); // Thanks to http://stackoverflow.com/a/6089413
}

template <typename T>
__device__ inline constexpr T Sqr(T v) {
    return v * v;
}

__host__ __device__ inline float AbsDot(glm::vec3 a, glm::vec3 b)
{
    return abs(dot(a, b));
}

__host__ __device__ inline float CosTheta(glm::vec3 w) { return w.z; }
__host__ __device__ inline float Cos2Theta(glm::vec3 w) { return w.z * w.z; }
__host__ __device__ inline float AbsCosTheta(glm::vec3 w) { return abs(w.z); }
__host__ __device__ inline float Sin2Theta(glm::vec3 w)
{
    return glm::max(0.f, 1.f - Cos2Theta(w));
}
__host__ __device__ inline float SinTheta(glm::vec3 w) { return sqrt(Sin2Theta(w)); }
__host__ __device__ inline float TanTheta(glm::vec3 w) { return SinTheta(w) / CosTheta(w); }

__host__ __device__ inline float Tan2Theta(glm::vec3 w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}

__host__ __device__ inline float CosPhi(glm::vec3 w)
{
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}
__host__ __device__ inline float SinPhi(glm::vec3 w)
{
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}
__host__ __device__ inline float Cos2Phi(glm::vec3 w) { return CosPhi(w) * CosPhi(w); }
__host__ __device__ inline float Sin2Phi(glm::vec3 w) { return SinPhi(w) * SinPhi(w); }
__device__ static bool Refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt)
{
    // Compute cos theta using Snell's law
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1)
        return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}

__device__ static glm::vec3 Faceforward(glm::vec3 n, glm::vec3 v)
{
    return (dot(n, v) < 0.f) ? -n : n;
}

__device__ static bool SameHemisphere(glm::vec3 w, glm::vec3 wp)
{
    return w.z * wp.z > 0;
}

__device__ static void coordinateSystem(glm::vec3& v1, glm::vec3& v2, glm::vec3& v3)
{
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

__device__ static glm::mat3 LocalToWorld(glm::vec3 nor)
{
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__device__ static glm::mat3 WorldToLocal(glm::vec3 nor)
{
    return transpose(LocalToWorld(nor));
}

__device__ static float DistanceSquared(glm::vec3 p1, glm::vec3 p2)
{
    return dot(p1 - p2, p1 - p2);
}

__device__ static glm::vec3 squareToDiskConcentric(glm::vec2 xi)
{
    // TODO
    glm::vec2 uOffset = xi * 2.f - 1.f;
    if (uOffset.x == 0 && uOffset.y == 0)
        return glm::vec3(0);
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = FOUR_PI * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = TWO_PI - FOUR_PI * (uOffset.x / uOffset.y);
    }
    return r * glm::vec3(cos(theta), sin(theta), 0);
}

__device__ static glm::vec3 squareToHemisphereCosine(glm::vec2 xi)
{
    float x = xi.x, y = xi.y;
    glm::vec3 ret;
    ret.z = sqrt(1 - x);
    float sinTheta = sqrt(x), phi = TWO_PI * y;
    ret.x = sinTheta * cos(phi);
    ret.y = sinTheta * sin(phi);
    return ret;
}

__device__ static float squareToHemisphereCosinePDF(glm::vec3 sample)
{
    // DONE
    return sample.z * INV_PI;
}

__device__ static glm::vec3 squareToSphereUniform(glm::vec2 sample)
{
    // TODO
    glm::vec3 ret;
    float c1 = sample.x, c2 = sample.y;
    ret.z = 1 - 2 * c1;
    float factor = sqrt(1 - ret.z * ret.z);
    ret.x = cos(TWO_PI * c2) * factor;
    ret.y = sin(TWO_PI * c2) * factor;
    return ret;
}

__device__ static float squareToSphereUniformPDF(glm::vec3 sample)
{
    // TODO
    return INV_FOUR_PI;
}
