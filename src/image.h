#pragma once

#include <glm/glm.hpp>

using namespace std;

class image {
private:
    int xSize;
    int ySize;
    glm::vec3 *pixels;

public:
    image(int x, int y);
    image(const std::string &baseFilename);
    ~image();
    int getWidth(){return xSize;};
    int getHeight(){return ySize;};
    glm::vec3* getPixel() { return pixels; };

    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
};
