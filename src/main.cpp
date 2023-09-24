#include "main.h"
#include "preview.h"
#include "application.h"
#include <cstring>

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


	//// GLFW main loop
	//mainLoop();
	auto& app = Application::getInstance();
	app.loadScene(sceneFilePath.c_str());
	app.run();

	return 0;
}
