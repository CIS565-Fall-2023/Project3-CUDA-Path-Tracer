#include "ObjLoader.h"
#include <iostream>

ObjLoader::ObjLoader() {}

ObjLoader::~ObjLoader() {}

bool ObjLoader::Load(const std::string& filename) {
    std::string warn;
    std::string err;
    std::string base_dir = filename.substr(0, filename.find_last_of("/\\") + 1); // Extract directory for .mtl files

    bool ret = tinyobj::LoadObj(&attrib_, &shapes_, &materials_, &warn, &err, filename.c_str(), base_dir.c_str());

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
    }

    return ret;
}

const std::vector<tinyobj::shape_t>& ObjLoader::GetShapes() const {
    return shapes_;
}

const std::vector<tinyobj::material_t>& ObjLoader::GetMaterials() const {
    return materials_;
}
