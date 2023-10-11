#include "main.h"
#include "preview.h"
#include <cstring>
#include <chrono>
#include <stb_image.h>
#include <OpenImageDenoise/oidn.h>
static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;
uint64_t sysTime;
uint64_t delta_t;

bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

OIDNDevice device;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];


	// Load scene file
	scene = new Scene(sceneFile);
	scene->buildBVH();
	scene->buildStacklessBVH();

	scene->CreateLights();

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();
	sysTime = time(nullptr);
	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, 1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();
	stbi_set_flip_vertically_on_load(1);
	scene->LoadAllTextures();

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT); // CPU or GPU if available
	oidnCommitDevice(device);
	// GLFW main loop
	mainLoop();

	oidnReleaseDevice(device);
	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);
#if DENOISE
	DrawGbuffer(16);
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			renderState->image[index] /= samples;
		}
	}
	samples = 1;
	OIDNBuffer colorBuf = oidnNewBuffer(device, width * height * 3 * sizeof(float));
	OIDNBuffer albedoBuf = oidnNewBuffer(device, width * height * 3 * sizeof(float));
	OIDNBuffer normalBuf = oidnNewBuffer(device, width * height * 3 * sizeof(float));
	OIDNFilter filter = oidnNewFilter(device, "RT");

	oidnSetFilterImage(filter, "color", colorBuf,
		OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // beauty
	oidnSetFilterImage(filter, "albedo", albedoBuf,
		OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // auxiliary
	oidnSetFilterImage(filter, "normal", normalBuf,
		OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // auxiliary
	oidnSetFilterImage(filter, "output", colorBuf,
		OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // denoised beauty
	oidnSetFilterBool(filter, "hdr", true); // beauty image is HDR
	oidnCommitFilter(filter);

	float* colorPtr = (float*)oidnGetBufferData(colorBuf);
	float* albedoPtr = (float*)oidnGetBufferData(albedoBuf);
	float* normalPtr = (float*)oidnGetBufferData(normalBuf);
	memcpy(colorPtr, renderState->image.data(), width * height * 3 * sizeof(float));
	memcpy(albedoPtr, renderState->albedo.data(), width * height * 3 * sizeof(float));
	memcpy(normalPtr, renderState->normal.data(), width * height * 3 * sizeof(float));

	oidnExecuteFilter(filter);
	const char* errorMessage;
	if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
		printf("Error: %s\n", errorMessage);

	memcpy(renderState->image.data(), colorPtr, width * height * 3 * sizeof(float));
	oidnReleaseBuffer(colorBuf);
	oidnReleaseBuffer(albedoBuf);
	oidnReleaseBuffer(normalBuf);
	oidnReleaseFilter(filter);
#endif

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
#if TONEMAPPING
			img.setPixel(width - 1 - x, y, util_postprocess_gamma(util_postprocess_ACESFilm(pix / samples)));
#else
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
#endif
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
#if DENOISE
	ss << "_denoised";
#endif
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cam.view.x = sin(phi) * sin(theta);
		cam.view.y = cos(theta);
		cam.view.z = cos(phi) * sin(theta);

		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		//cam.position = cameraPosition;
		camchanged = false;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		pathtraceFree(scene);
		pathtraceInit(scene);
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		pathtrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		pathtraceFree(scene);
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_Q:
			saveImage();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		}
		
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta += (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}
