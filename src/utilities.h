#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"

class GuiDataContainer
{
public:
    GuiDataContainer() : TracedDepth(0) {}
    int TracedDepth;
};

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/
}

INILNE CPU_GPU void CoordinateSystem(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent)
{
	glm::vec3 up = glm::abs(normal.z) < 0.999f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
	tangent = glm::normalize(glm::cross(up, normal));
	bitangent = glm::cross(normal, tangent);
}

INILNE CPU_GPU glm::mat3 LocalToWorld(const::glm::vec3& nor)
{
	glm::vec3 tan, bit;
	CoordinateSystem(nor, tan, bit);
	return glm::mat3(tan, bit, nor);
}
INILNE CPU_GPU glm::mat3 WorldToLocal(const::glm::vec3& nor)
{
	return glm::transpose(LocalToWorld(nor));
}

template<typename T>
CPU_GPU T BarycentricInterpolation(const T& c0, const T& c1, const T& c2, const glm::vec2& uv)
{
	return (1.f - uv.x - uv.y) * c0 + uv.x * c1 + uv.y * c2;
}