#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

namespace Utils
{
    __host__ __device__ float luminance(glm::vec3 color)
    {
        return glm::dot(color, glm::vec3(0.2126, 0.7152, 0.0722));
    }
}