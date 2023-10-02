#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "toggles.h"

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define PI_OVER_FOUR      0.7853981633974483096156608458198757210492f
#define PI_OVER_TWO       1.5707963267948966192313216916397514420985f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

class GuiDataContainer
{
public:
    GuiDataContainer() {}

    int tracedDepth{ 0 };

    bool sortByMaterial{ false };
#if FIRST_BOUNCE_CACHE
    bool firstBounceCache{ true };
#endif
    bool russianRoulette{ true };
    bool useBvh{ true };
    bool denoising{ true };
    int denoiseInterval{ 3 };

    float lensRadius{ 0.f };
    float focusDistance{ 0.f };

    bool showAlbedo{ false };
    bool showNormals{ false };
};

namespace Utils {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
    extern uint64_t timeSinceEpochMillisec();
    extern bool filePathHasExtension(const std::string& filePath, const std::string& ext);
}