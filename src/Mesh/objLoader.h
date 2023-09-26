#pragma once

#include <string>
#include <vector>
#include "tiny_obj_loader.h"

class ObjLoader
{
public:
	ObjLoader();
	~ObjLoader();

	bool Load(const std::string& filename);
	const std::vector<tinyobj::shape_t>& GetShapes() const;
	const std::vector<tinyobj::material_t>& GetMaterials() const;

private:
	tinyobj::attrib_t attrib_;
	std::vector<tinyobj::shape_t> shapes_;
	std::vector<tinyobj::material_t> materials_;
};

