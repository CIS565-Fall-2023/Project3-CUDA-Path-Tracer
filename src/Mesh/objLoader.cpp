#pragma once

#include "ObjLoader.h"
#include <iostream>


bool ObjLoader::Load(const std::string filename) {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = ""; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        return false;
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    attrib = reader.GetAttrib();
    shapes = reader.GetShapes();
    materials = reader.GetMaterials();

    return true;
}