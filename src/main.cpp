#include "main.h"
#include "preview.h"
#include <cstring>
#include <glm/gtx/string_cast.hpp>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static double lastX;
static double lastY;

static bool camChanged = true;

static glm::vec3 camOriginalPos;
static glm::vec3 camOriginalLookAt;

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

	camOriginalPos = cam.position;
	camOriginalLookAt = cam.lookAt;

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
	img.savePNG(filename, guiData->showNormals);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda(bool reset, GuiDataContainer* guiData) {
	if (camChanged) {
		iteration = 0;

		Camera& cam = renderState->camera;

		cam.view = glm::normalize(cam.lookAt - cam.position);
		cam.right = glm::normalize(glm::cross(cam.view, glm::vec3(0, 1, 0)));
		cam.up = glm::normalize(glm::cross(cam.right, cam.view));

		Pathtracer::onCamChanged();

		camChanged = false;
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
			renderState = &scene->state;
			cam.position = camOriginalPos;
			cam.lookAt = camOriginalLookAt;
			camChanged = true;
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

	cam.move(moveSpeed.x * cam.right
		+ moveSpeed.y * glm::vec3(0, 1, 0)
		+ moveSpeed.z * cam.view);

	camChanged = true;
}

const Camera& getCamera()
{
	return renderState->camera;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	
	if (leftMousePressed) {
		float sensitivity = 1.5f;

		Camera& cam = scene->state.camera;

		float dTheta = (xpos - lastX) * sensitivity / width;
		float dPhi = -(ypos - lastY) * sensitivity / height;

		glm::vec3 diff = cam.lookAt - cam.position;
		float theta = atan2f(diff.z, diff.x) + dTheta;
		float phi = atan2f(diff.y, sqrtf(diff.x * diff.x + diff.z * diff.z)) + dPhi;
		phi = glm::clamp(phi, -PI_OVER_TWO, PI_OVER_TWO);

		float newX = cosf(theta) * cosf(phi);
		float newY = sinf(phi);
		float newZ = sinf(theta) * cosf(phi);
		cam.lookAt = cam.position + glm::vec3(newX, newY, newZ) * glm::length(diff);

		camChanged = true;
	}

	lastX = xpos;
	lastY = ypos;
}
