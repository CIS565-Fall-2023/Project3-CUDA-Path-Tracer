#include "main.h"
#include "application.h"
#include <cstring>
#include <glm/gtx/intersect.hpp>
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
	sceneFilePath += "refract.txt";

	glm::vec3 wi(0, 1, 0);
	glm::vec3 n(0, 1, 0); 
	float idxOfRrefract = 1/1.5;
	glm::vec3 ref = glm::refract(wi, n, idxOfRrefract);
	float R0 = pow((1 - idxOfRrefract) / (1 + idxOfRrefract), 2.);
	float fresnel = R0 + (1 - R0) * pow(1 - glm::dot(n, wi), 5);//how much is reflected
	cout << "fresnel: " << fresnel << endl;
	cout << "wi: " << displayVec3(wi) << " normal: " << displayVec3(n) << " refract: " << displayVec3(ref) << endl;
	cout << "refract length: " << glm::length(ref) << endl;


	wi = glm::vec3(0, -1, 0);
	ref = glm::refract(wi, n, idxOfRrefract);
	cout << (glm::isnan(ref).x ? "true" : "false") << endl;
	fresnel = R0 + (1 - R0) * pow(1 - glm::dot(n, wi), 5);//how much is reflected
	cout << "fresnel: " << fresnel << endl;
	cout << "wi: " << displayVec3(wi) << " normal: " << displayVec3(n) << " refract: " << displayVec3(ref) << endl;
	cout << "refract length: " << glm::length(ref) << endl;

	wi = glm::normalize(glm::vec3(0.5, 1, 0));
	ref = glm::refract(wi, n, idxOfRrefract);
	cout << (glm::isnan(ref).x ? "true" : "false") << endl;
	fresnel = R0 + (1 - R0) * pow(1 - glm::dot(n, wi), 5);//how much is reflected
	cout << "fresnel: " << fresnel << endl;
	cout << "wi: " << displayVec3(wi) << " normal: " << displayVec3(n) << " refract: " << displayVec3(ref) << endl;
	cout << "refract length: " << glm::length(ref) << endl;

	wi = glm::normalize(glm::vec3(0.5, -1, 0));
	ref = glm::refract(wi, n, idxOfRrefract);
	cout << (glm::isnan(ref).x ? "true" : "false") << endl;
	fresnel = R0 + (1 - R0) * pow(1 - glm::dot(n, wi), 5);//how much is reflected
	cout << "fresnel: " << fresnel << endl;
	cout << "wi: " << displayVec3(wi) << " normal: " << displayVec3(n) << " refract: " << displayVec3(ref) << endl;
	cout << "refract length: " << glm::length(ref) << endl;

	wi = glm::normalize(glm::vec3(5, 1, 0));
	ref = glm::refract(wi, n, idxOfRrefract);
	cout << (glm::isnan(ref).x ? "true" : "false") << endl;
	fresnel = R0 + (1 - R0) * pow(1 - glm::dot(n, wi), 5);//how much is reflected
	cout << "fresnel: " << fresnel << endl;
	cout << "wi: " << displayVec3(wi) << " normal: " << displayVec3(n) << " refract: " << displayVec3(ref) << endl;

	wi = glm::normalize(glm::vec3(5, -1, 0));
	ref = glm::refract(wi, n, idxOfRrefract);
	cout << (isnan(ref).x?"true":"false") << endl;
	fresnel = R0 + (1 - R0) * pow(1 - glm::dot(n, wi), 5);//how much is reflected
	cout << "fresnel: " << fresnel << endl;
	cout << "wi: " << displayVec3(wi) << " normal: " << displayVec3(n) << " refract: " << displayVec3(ref) << endl;

	auto& app = Application::getInstance();
	app.loadScene(sceneFilePath.c_str());
	app.run();

	return 0;
}
