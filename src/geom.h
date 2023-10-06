#pragma once

#include "bound.h"
#include "glm/glm.hpp"


enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE
};

class Geom {
public:
    GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    virtual Bound getObjectSpaceBounds() const = 0;
    Bound getBounds() const;
};

class Cube : public Geom {
public:
    Cube();
    Bound getObjectSpaceBounds() const override;
};


class Sphere : public Geom {
public:
    Sphere();
    Bound getObjectSpaceBounds() const override;
};

class Triangle : public Geom {
public:
    glm::vec3 v0, v1, v2;

    Triangle();

    Bound getObjectSpaceBounds() const override;
};

