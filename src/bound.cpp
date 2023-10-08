#include "bound.h"

Bound::Bound()
    : pMin(glm::vec3(FLT_MAX)), pMax(glm::vec3(-FLT_MAX))
{}

Bound::Bound(const glm::vec3& p1, const glm::vec3& p2)
    : pMin(glm::min(p1, p2)), pMax(glm::max(p1, p2))
{}

int Bound::getLongestAxis() const {
    glm::vec3 diff = pMax - pMin;

    if (diff.x > diff.y && diff.x > diff.z) {
        return 0;
    }
    else if (diff.y > diff.z) {
        return 1;
    }
    else {
        return 2;
    }
}

glm::vec3 Bound::offset(const glm::vec3& point) const {
    glm::vec3 offset = point - pMin;

    if (pMax.x > pMin.x) {
        offset.x /= pMax.x - pMin.x;
    }
    if (pMax.y > pMin.y) {
        offset.y /= pMax.y - pMin.y;
    }
    if (pMax.z > pMin.z) {
        offset.z /= pMax.z - pMin.z;
    }

    return offset;
}

float Bound::computeBoxSurfaceArea() const {
    glm::vec3 diff = pMax - pMin;
    return 2.0f * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z);
}

Bound Bound::unionBound(const glm::vec3& p) {
    return Bound(glm::min(pMin, p), glm::max(pMax, p));
}

Bound Bound::unionBound(const Bound& otherBound) {
    return Bound(glm::min(pMin, otherBound.pMin), glm::max(pMax, otherBound.pMax));
}

const glm::vec3& Bound::operator[](int i) const
{
    if (i == 0) return pMin;
    else return pMax;
}

glm::vec3& Bound::operator[](int i)
{
    if (i == 0) return pMin;
    else return pMax;
}
