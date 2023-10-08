#include <iostream>
#include "config.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

SceneConfig::SceneConfig(string filename) {
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
            if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "IBL") == 0) {
                loadEnvironmentMap();
            }
        }
    }
}

int SceneConfig::loadCamera() {
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

int SceneConfig::loadEnvironmentMap()
{
    cout << "Loading environment map..." << endl;

    string line;
    utilityCore::safeGetline(fp_in, line);

    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            cout << tokens[1] << endl;

            float* exr_pixels;
            int width, height;
            const char* err = nullptr;
            //const char* filename = "img/colorful_studio_4k.exr";

            int ret;
            const char * file = tokens[1].c_str();
            // Load the EXR image using tinyexr
            if ((ret = LoadEXR(&exr_pixels, &width, &height, file, &err))) {
                // Handle error loading EXR image
                if (err) {
                    fprintf(stderr, "ERR : %s\n", err);
                    FreeEXRErrorMessage(err); // release memory of error message.
                    assert(0);
                }
            }
            EXRVersion exr_version;
            if ((ret = ParseEXRVersionFromFile(&exr_version, file))) {
                fprintf(stderr, "ERR : %s\n", err);
                assert(0);
            }

            if (exr_version.multipart) {
                assert(0);
            }

            EXRHeader exr_header;
            InitEXRHeader(&exr_header);

            if ((ret = ParseEXRHeaderFromFile(&exr_header, &exr_version, file, &err))) {
                fprintf(stderr, "ERR : %s\n", err);
                assert(0);
            }

            EXRImage exr_image;
            InitEXRImage(&exr_image);

            ret = LoadEXRImageFromFile(&exr_image, &exr_header, file, &err);
            if (ret != 0) {
                fprintf(stderr, "Load EXR err: %s\n", err);
                FreeEXRHeader(&exr_header);
                FreeEXRErrorMessage(err); // free's buffer for an error message
                //return ret;
            }
            printf("Loaded!\n");
            env_map.width = exr_image.width;
            env_map.height = exr_image.height;
            env_map.nrChannels = exr_image.num_channels;
            env_map.data.assign(env_map.width * env_map.height * env_map.nrChannels, 0.0f);
            auto images = exr_image.images;
            auto image_r = reinterpret_cast<float*>(images[0]);
            auto image_g = reinterpret_cast<float*>(images[1]);
            auto image_b = reinterpret_cast<float*>(images[2]);
            for (size_t i = 0; i < width* height; i++)
            {
                auto pixel = glm::vec3(image_r[i], image_g[i], image_b[i]);
                //printf("pixel: %X %X %X\n", pixel[0], pixel[1], pixel[2]);
                //printf("pixel: %d %d %d\n\n", pixel[0], pixel[1], pixel[2]);
                env_map.data[i * 3 + 0] = glm::clamp(powf(pixel[2], 0.55f) * 255, 0.0f, 255.0f);
                env_map.data[i * 3 + 1] = glm::clamp(powf(pixel[1], 0.55f) * 255, 0.0f, 255.0f);
                env_map.data[i * 3 + 2] = glm::clamp(powf(pixel[0], 0.55f) * 255, 0.0f, 255.0f);
                //env_map.data[i * 3 + 1] = 0;
                //env_map.data[i * 3 + 2] = 0;
            }
            has_env_map = true;
        }
        utilityCore::safeGetline(fp_in, line);
    }
    return 0;
}
