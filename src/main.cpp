#include "main.h"
#include "application.h"
#include <cstring>
#include <glm/gtx/intersect.hpp>
#include "cuda_runtime.h"
//#include "interactions.h"
//-------------------------------
//-------------MAIN--------------
//-------------------------------

string displayVec3(glm::vec3 v) {
	std::stringstream sstr;
	sstr <<"[ " << v.x << "," << v.y << "," << v.z << "]";
	return sstr.str();
}

int main(int argc, char** argv) {
	//startTimeString = currentTimeString();

	std::cout << SCENE_PATH << std::endl;

	//if (argc < 2) {
	//	printf("Usage: %s SCENEFILE.txt\n", argv[0]);
	//	return 1;
	//}

	string sceneFilePath = SCENE_PATH;
	sceneFilePath += "single_cube.txt";

	auto& app = Application::getInstance();

	app.loadScene(sceneFilePath.c_str());
	app.run();

	return 0;
}
