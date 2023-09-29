#pragma once

#include <glm/glm.hpp>
#include <thrust/random.h>
#include "utilities.h"

enum BSDFType {
    UNIMPLEMENTED = -1,
    DIFFUSE = 0,
    SPECULAR = 1,
    REFRACTIVE = 2,
    MICROFACET = 3,
    EMISSIVE = 4
};

class BSDFStruct {
public:
    glm::vec3 reflectance;
    glm::vec3 emissiveFactor;
    float strength;
    BSDFType bsdfType;
    int diffuseTextureID;
    int roughtnessTextureID;
};