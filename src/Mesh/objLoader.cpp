#include "ObjLoader.h"
#include <iostream>


bool ObjLoader::Load(const std::string filename) {
    //tinyobj::ObjReaderConfig reader_config;
    //// reader_config.mtl_search_path = ""; // Path to material files

    //tinyobj::ObjReader reader;

    //if (!reader.ParseFromFile(filename, reader_config)) {
    //    if (!reader.Error().empty()) {
    //        std::cerr << "TinyObjReader: " << reader.Error();
    //    }
    //    return false;
    //}

    //if (!reader.Warning().empty()) {
    //    std::cout << "TinyObjReader: " << reader.Warning();
    //}

    //auto& attrib = reader.GetAttrib();
    //auto& shapes = reader.GetShapes();
    //// auto& materials = reader.GetMaterials();

    //return true;

    std::string inputfile = "../scenes/bunny.obj";

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(1);
    }

    return true;
}