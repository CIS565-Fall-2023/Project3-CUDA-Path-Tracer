#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
using namespace std;

class image {
public:
    int xSize;
    int ySize;
    uchar4 *pixels;

public:
    image(int x, int y);
    ~image();
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
};
