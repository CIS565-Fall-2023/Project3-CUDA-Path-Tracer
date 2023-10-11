#pragma once

#include <string>
#include <vector>
#include "tiny_obj_loader.h"

class ObjLoader
{
public:
	bool Load(const std::string filename);

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
};

