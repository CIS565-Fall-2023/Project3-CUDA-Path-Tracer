#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#include "tiny_obj_loader.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

bool Scene::loadObj(Geom& geom, const string& objFile) {
    tinyobj::attrib_t attribs;
    vector<tinyobj::shape_t> shapes;
    vector<tinyobj::material_t> materials;
    string warn, err;

    bool res = tinyobj::LoadObj(&attribs, &shapes, &materials, &warn, &err, objFile.c_str());
    if (!res) {
        cout << "Loading obj file failed!" << endl;
        cout << "Warn: " << warn << endl;
		cout << "Error: " << err << endl;
		return res;
	}

    cout << "Tiny obj load success!" << endl;
    
    geom.triangleStart = triangles.size();
    for (auto& shape : shapes) {
        auto& mesh = shape.mesh;
        for (int i = 0; i < mesh.indices.size(); i += 3) {
            Triangle triangle;
			triangle.position[0] = glm::vec3(
					attribs.vertices[3 * mesh.indices[i].vertex_index],
					attribs.vertices[3 * mesh.indices[i].vertex_index + 1],
					attribs.vertices[3 * mesh.indices[i].vertex_index + 2]);
			triangle.position[1] = glm::vec3(
					attribs.vertices[3 * mesh.indices[i + 1].vertex_index],
					attribs.vertices[3 * mesh.indices[i + 1].vertex_index + 1],
					attribs.vertices[3 * mesh.indices[i + 1].vertex_index + 2]);
			triangle.position[2] = glm::vec3(
					attribs.vertices[3 * mesh.indices[i + 2].vertex_index],
					attribs.vertices[3 * mesh.indices[i + 2].vertex_index + 1],
					attribs.vertices[3 * mesh.indices[i + 2].vertex_index + 2]);
			triangle.normal[0] = glm::vec3(
					attribs.normals[3 * mesh.indices[i].normal_index],
					attribs.normals[3 * mesh.indices[i].normal_index + 1],
					attribs.normals[3 * mesh.indices[i].normal_index + 2]);
			triangle.normal[1] = glm::vec3(
					attribs.normals[3 * mesh.indices[i + 1].normal_index],
					attribs.normals[3 * mesh.indices[i + 1].normal_index + 1],
					attribs.normals[3 * mesh.indices[i + 1].normal_index + 2]);
			triangle.normal[2] = glm::vec3(
					attribs.normals[3 * mesh.indices[i + 2].normal_index],
					attribs.normals[3 * mesh.indices[i + 2].normal_index + 1],
					attribs.normals[3 * mesh.indices[i + 2].normal_index + 2]);
			//triangle.t0 = glm::vec2(
			//		attribs.texcoords[2 * mesh.indices[i]],
			//		attribs.texcoords[2 * mesh.indices[i] + 1]);
			//triangle.t1 = glm::vec2(
			//		attribs.texcoords[2 * mesh.indices[i + 1]],
			//		attribs.texcoords[2 * mesh.indices[i + 1] + 1]);
			//triangle.t2 = glm::vec2(
			//		attribs.texcoords[2 * mesh.indices[i + 2]],
			//		attribs.texcoords[2 * mesh.indices[i + 2] + 1]);
            triangles.push_back(triangle);
        }
    }

    geom.triangleEnd = triangles.size(); // not include

    return res;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        newGeom.geomId = id;

        string line;

        LightType lightType = LightType::NONE;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(line.c_str(), "obj") == 0) {
                cout << "Creating new mesh..." << endl;
                newGeom.type = MESH;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "LIGHTTYPE") == 0) {
                lightType = (LightType)atoi(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "OBJFILE") == 0 && newGeom.type == GeomType::MESH) {
				// tiny obj load
                loadObj(newGeom, tokens[1]);
			}

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);


        // TODO: calculate bounding box
        newGeom.minPoint = glm::vec3(newGeom.transform * glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f));
        newGeom.maxPoint = glm::vec3(newGeom.transform * glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));

        for (int i = 0; i < 3; ++i) {
            if (newGeom.minPoint[i] > newGeom.maxPoint[i]) {
                std::swap(newGeom.minPoint[i], newGeom.maxPoint[i]);
            }
        }

        geoms.push_back(newGeom);

        // light
        if (lightType != LightType::NONE) {
            cout << "Loading Geom " << id << " as Light." << endl;
            Light light;
            light.geom = newGeom;
            light.lightType = lightType;
            // TODO: spot light
            lights.push_back(light);
        }

        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "FOCAL") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "LENS_RADIUS") == 0) {
			camera.lensRadius = atof(tokens[1].c_str());
		} 

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;
        newMaterial.roughness = 0.0f; //default

        //load static properties
        string line;
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.reflectivity = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.refractivity = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
                newMaterial.roughness = atof(tokens[1].c_str());
            }
            utilityCore::safeGetline(fp_in, line);
        }

        // judge material type
        newMaterial.type = judgeMaterialType(newMaterial.reflectivity, 
            newMaterial.refractivity, newMaterial.roughness);

        materials.push_back(newMaterial);
        return 1;
    }
}

MaterialType Scene::judgeMaterialType(float reflectivity, float refractivity, float roughness) {
    if (roughness == 0.0f) {
        if (reflectivity == 1.0f && refractivity == 1.0f) {
            // perfect mirror
            return MaterialType::SPEC_FRESNEL;
        }
        else if (reflectivity == 1.0f) {
            // perfect specular reflect
            return MaterialType::SPEC_REFL;
        }
        else if (refractivity == 1.0f) {
            // perfect specular transmission
            return MaterialType::SPEC_TRANS;
        }
        else if (reflectivity == 0.0f && refractivity == 0.0f) {
            // diffuse
			return MaterialType::DIFFUSE;
        }
    }
    else {
        if (reflectivity > 0.0f) {
            // microfacet
            //TODO: expand microfacet material type
            return MaterialType::MICROFACET;
        }
    }

    return MaterialType::DIFFUSE;
}