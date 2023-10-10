#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define USE_BVH 1
#define NUM_BVHBINS 8

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define INV_PI            0.3183098861837906715377675267450287240689f
#define HALF_PI           1.5707963267948966192313216916397514420985f
#define QUARTER_PI        0.7853981633974483096156608458198757210493f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

#define LUMINANCE(color) (0.2126 * (color).r + 0.7152 * (color).g + 0.0722 * (color).b)
#define FLOATMAX(a, b) ((a) < (b) ? (b) : (a))

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0), sortMaterial(false), cacheFirstBounce(false), fbCached(false), lensRadius(0.f), focalDistance(0.f) {}
    int TracedDepth;
    bool sortMaterial;
    bool cacheFirstBounce;
    bool fbCached;
    float lensRadius;
    float focalDistance;
};

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
    extern std::string extractFilename(const std::string& filePath);
    extern bool matchFileExtension(const std::string& filePath, std::string extension);
}
