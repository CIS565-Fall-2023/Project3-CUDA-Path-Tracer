#include "main.h"
#include "preview.h"
#include "application.h"
#include <cstring>
#include <glm/gtx/intersect.hpp>
//#include "interactions.h"
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	//startTimeString = currentTimeString();

	std::cout << SCENE_PATH << std::endl;

	//if (argc < 2) {
	//	printf("Usage: %s SCENEFILE.txt\n", argv[0]);
	//	return 1;
	//}

	string sceneFilePath = SCENE_PATH;
	sceneFilePath += "cornell.txt";

	glm::vec3 pts[] = { glm::vec3(0,0,1) , glm::vec3(1,0,0) , glm::vec3(0,1,0) };
	glm::vec3 orig = glm::vec3(0.f);
	glm::vec3 dir = glm::normalize(glm::vec3(1, 1, 1));
	glm::vec3 hit;
	if (glm::intersectRayTriangle(orig, dir, pts[0], pts[1], pts[2], hit)) {
		std::cout << hit[0] << "," << hit[1] << "," << hit[2] << std::endl;
	}
	else {
		std::cout << "not hit" << std::endl;
	}
	glm::vec3 c;
	if (glm::intersectRayTriangle(glm::vec3(0.f,2,0), glm::vec3(0, 0, 1), glm::vec3(5, 1, 3), glm::vec3(-5, 1, 3), glm::vec3(0, 5, 3), c)) {
		std::cout << c.x << "," << c.y << "," << c.z << std::endl;
	}else {
		std::cout << "not hit" << std::endl;
	}
	//// GLFW main loop
	//mainLoop();
	auto& app = Application::getInstance();
	app.loadScene(sceneFilePath.c_str());
	app.run();

	return 0;
}
