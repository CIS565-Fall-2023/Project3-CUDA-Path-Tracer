#pragma once

#include "glm/glm.hpp"
#include "bound.h"

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE
};

class Geom {
public:
    int type;
    int materialid;
    glm::vec3 translation, rotation, scale;
    glm::mat4 transform, inverseTransform, invTranspose;

    // Triangle properties
    glm::vec3 v0, v1, v2;  // Triangle vertices (in world space)
    glm::vec3 n0, n1, n2;  // Normals at each vertex (in world space)
    glm::vec2 uv0, uv1, uv2;  // UVs at each vertex

    // Constructor
    Geom(GeomType type, const int& materialid, const glm::vec3& translation,
        const glm::vec3& rotation, const glm::vec3& scale, const glm::mat4& transform,
        const glm::mat4& inverseTransform, const glm::mat4& invTranspose);

    void setVertices(const glm::vec3& vertex0, const glm::vec3& vertex1, const glm::vec3& vertex2);
    void setNormals(const glm::vec3& normal0, const glm::vec3& normal1, const glm::vec3& normal2);
    void setUVs(const glm::vec2& uv0, const glm::vec2& uv1, const glm::vec2& uv2);

    Bound getWorldBounds() const;
};