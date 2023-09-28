#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <stb_image.h>
#include "image.h"

image::image(int x, int y) :
        xSize(x),
        ySize(y),
        pixels(new uchar4[x * y]) {
}

image::~image() {
    delete pixels;
}

void image::savePNG(const std::string &baseFilename) {
    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 4, pixels, xSize * 4);
    std::cout << "Saved " << filename << "." << std::endl;
}

void image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}
