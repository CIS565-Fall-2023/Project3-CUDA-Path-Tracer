#pragma once 

#include <glm/glm.hpp>

class Bound {
public:
    glm::vec3 pMin, pMax;

    Bound();
    Bound(const glm::vec3& p1, const glm::vec3& p2);
    int getLongestAxis() const;
    glm::vec3 offset(const glm::vec3& point) const;
    float computeBoxSurfaceArea() const;
    Bound unionBound(const glm::vec3& p);
    Bound unionBound(const Bound& otherBound);

    // operator
    const glm::vec3& operator[](int i) const;
    glm::vec3& operator[](int i);
};