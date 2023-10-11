#include "geom.h"

Geom::Geom(GeomType type, const int& materialid, const glm::vec3& translation,
    const glm::vec3& rotation, const glm::vec3& scale, const glm::mat4& transform,
    const glm::mat4& inverseTransform, const glm::mat4& invTranspose)
    : type(type), materialid(materialid), translation(translation),
    rotation(rotation), scale(scale), transform(transform), inverseTransform(inverseTransform),
    invTranspose(invTranspose)
{
    v0 = v1 = v2 = glm::vec3(0);
    n0 = n1 = n2 = glm::vec3(0);
    uv0 = uv1 = uv2 = glm::vec2(0);
}

Bound Geom::getWorldBounds() const {
    glm::vec3 objectSpaceMin;
    glm::vec3 objectSpaceMax;

    // Get the object-space bounding box
    switch (type) {
        case TRIANGLE:
            objectSpaceMin = glm::min(v0, glm::min(v1, v2));
            objectSpaceMax = glm::max(v0, glm::max(v1, v2));
            break;
        case SPHERE:
            objectSpaceMin = glm::vec3(-scale.x);
            objectSpaceMax = glm::vec3(scale.x);
            break;
        case CUBE:
            objectSpaceMin = -0.5f * scale;
            objectSpaceMax = 0.5f * scale;
            break;
        default:
            objectSpaceMin = glm::vec3(0);
            objectSpaceMax = glm::vec3(0);
            break;
        }

    // Transform object-space bounding box corners to world space
    glm::vec3 corners[8] = {
        objectSpaceMin,
        glm::vec3(objectSpaceMax.x, objectSpaceMin.y, objectSpaceMin.z),
        glm::vec3(objectSpaceMin.x, objectSpaceMax.y, objectSpaceMin.z),
        glm::vec3(objectSpaceMin.x, objectSpaceMin.y, objectSpaceMax.z),
        glm::vec3(objectSpaceMax.x, objectSpaceMax.y, objectSpaceMin.z),
        glm::vec3(objectSpaceMax.x, objectSpaceMin.y, objectSpaceMax.z),
        glm::vec3(objectSpaceMin.x, objectSpaceMax.y, objectSpaceMax.z),
        objectSpaceMax
    };

    Bound bound;

    for (int i = 0; i < 8; i++) {
        glm::vec3 worldCorner = glm::vec3(transform * glm::vec4(corners[i], 1.0f));
        bound = bound.unionBound(worldCorner);
    }

    return bound;
}

// Method to set triangle vertices and transform them to world space
void Geom::setVertices(const glm::vec3 & vertex0, const glm::vec3 & vertex1, const glm::vec3 & vertex2) {
    v0 = glm::vec3(transform * glm::vec4(vertex0, 1.0f));
    v1 = glm::vec3(transform * glm::vec4(vertex1, 1.0f));
    v2 = glm::vec3(transform * glm::vec4(vertex2, 1.0f));
}

// Method to set triangle normals and transform them to world space
void Geom::setNormals(const glm::vec3& normal0, const glm::vec3& normal1, const glm::vec3& normal2) {
    n0 = glm::normalize(glm::vec3(invTranspose * glm::vec4(normal0, 0.0f)));
    n1 = glm::normalize(glm::vec3(invTranspose * glm::vec4(normal1, 0.0f)));
    n2 = glm::normalize(glm::vec3(invTranspose * glm::vec4(normal2, 0.0f)));
}

void Geom::setUVs(const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2) {
    this->uv0 = uv0;
    this->uv1 = uv1;
    this->uv2 = uv2;
}
