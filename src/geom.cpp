#include "geom.h"

Bound Geom::getBounds() const {
    Bound objectBounds = getObjectSpaceBounds();

    // Transform object-space bounding box corners to world space
    glm::vec3 corners[8] = {
        objectBounds.pMin,
        glm::vec3(objectBounds.pMax.x, objectBounds.pMin.y, objectBounds.pMin.z),
        glm::vec3(objectBounds.pMin.x, objectBounds.pMax.y, objectBounds.pMin.z),
        glm::vec3(objectBounds.pMin.x, objectBounds.pMin.y, objectBounds.pMax.z),
        glm::vec3(objectBounds.pMax.x, objectBounds.pMax.y, objectBounds.pMin.z),
        glm::vec3(objectBounds.pMax.x, objectBounds.pMin.y, objectBounds.pMax.z),
        glm::vec3(objectBounds.pMin.x, objectBounds.pMax.y, objectBounds.pMax.z),
        objectBounds.pMax
    };

    Bound worldBounds;

    for (int i = 0; i < 8; i++) {
        glm::vec3 worldCorner = glm::vec3(transform * glm::vec4(corners[i], 1.0f));
        worldBounds = worldBounds.unionBound(worldCorner);
    }

    return worldBounds;
}

Cube::Cube()
{
    type = CUBE;
}

Bound Cube::getObjectSpaceBounds() const {
    glm::vec3 objectSpaceMin = -0.5f * scale;
    glm::vec3 objectSpaceMax = 0.5f * scale;
    return Bound(objectSpaceMin, objectSpaceMax);
}

Sphere::Sphere()
{
    type = SPHERE;
}

Bound Sphere::getObjectSpaceBounds() const {
    glm::vec3 objectSpaceMin = glm::vec3(-scale.x);
    glm::vec3 objectSpaceMax = glm::vec3(scale.x);
    return Bound(objectSpaceMin, objectSpaceMax);
}


Triangle::Triangle() 
{
	type = TRIANGLE;
}


Bound Triangle::getObjectSpaceBounds() const {
    glm::vec3 objectSpaceMin = glm::min(v0, glm::min(v1, v2));
    glm::vec3 objectSpaceMax = glm::max(v0, glm::max(v1, v2));

    return Bound(objectSpaceMin, objectSpaceMax);
}