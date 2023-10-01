#include "main.h"
#include "preview.h"
#include <cstring>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
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

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	guiData->lensRadius = cam.lensRadius;
	guiData->focusDistance = cam.focusDistance;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();

	// Initialize ImGui Data
	InitImguiData(guiData);
	Pathtracer::InitDataContainer(guiData);

	// GLFW main loop
	mainLoop();

	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index] / samples;

			if (guiData->showNormals)
			{
				pix = (pix + 1.f) / 2.f;
			}

			img.setPixel(width - 1 - x, y, pix);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda(bool reset, GuiDataContainer* guiData) {
	if (camchanged) {
		iteration = 0;

		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;

		Pathtracer::onCamChanged();

		camchanged = false;
	}

	if (reset)
	{
		iteration = 0;

		Camera& cam = renderState->camera;
		cam.lensRadius = guiData->lensRadius;
		cam.focusDistance = guiData->focusDistance;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		Pathtracer::free();
		Pathtracer::init(scene);
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		Pathtracer::pathtrace(pbo_dptr, frame, iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		Pathtracer::free();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

static glm::ivec3 camMovement = glm::ivec3(0);
static int speed = 0;
static const float baseSpeed = 0.3f;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		Camera& cam = renderState->camera;

		switch (key)
		{
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_F:
			saveImage();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			renderState = &scene->state;
			cam.lookAt = ogLookAt;
			break;
		case GLFW_KEY_W:
			++camMovement.z;
			break;
		case GLFW_KEY_S:
			--camMovement.z;
			break;
		case GLFW_KEY_D:
			++camMovement.x;
			break;
		case GLFW_KEY_A:
			--camMovement.x;
			break;
		case GLFW_KEY_E:
			++camMovement.y;
			break;
		case GLFW_KEY_Q:
			--camMovement.y;
			break;
		case GLFW_KEY_LEFT_SHIFT:
			++speed;
			break;
		case GLFW_KEY_LEFT_ALT:
			--speed;
			break;
		}
	}

	if (action == GLFW_RELEASE)
	{
		switch (key)
		{
		case GLFW_KEY_W:
			--camMovement.z;
			break;
		case GLFW_KEY_S:
			++camMovement.z;
			break;
		case GLFW_KEY_D:
			--camMovement.x;
			break;
		case GLFW_KEY_A:
			++camMovement.x;
			break;
		case GLFW_KEY_E:
			--camMovement.y;
			break;
		case GLFW_KEY_Q:
			++camMovement.y;
			break;
		case GLFW_KEY_LEFT_SHIFT:
			--speed;
			break;
		case GLFW_KEY_LEFT_ALT:
			++speed;
			break;
		}
	}
}

void moveCam()
{
	if (camMovement.x == 0 && camMovement.y == 0 && camMovement.z == 0)
	{
		return;
	}

	Camera& cam = renderState->camera;

	glm::vec3 moveSpeed = glm::vec3(camMovement) * baseSpeed;
	if (speed == -1)
	{
		moveSpeed *= 0.1f;
	}
	else if (speed == 1)
	{
		moveSpeed *= 10.f;
	}

	cam.lookAt += moveSpeed.x * cam.right 
		+ moveSpeed.y * glm::vec3(0, 1, 0)
		+ moveSpeed.z * cam.view;

	camchanged = true;
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
		float sensitivity = 1.5f;
		phi -= (xpos - lastX) * sensitivity / width;
		theta -= (ypos - lastY) * sensitivity / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) * 10.f / height;
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
