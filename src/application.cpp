#include "application.h"

const GLuint VS_POS_HANDLE = 0;
const GLuint VS_UV_HANDLE = 1;

static float zoom, theta, phi;
static glm::vec3 ogLookAt;

Application::Application()
	:m_width(100),m_height(100)
{
	init();
}

bool Application::init()
{
	m_guiData = make_unique<GuiDataContainer>();
	m_tracer = make_unique<PathTracer>();
	glfwSetErrorCallback([](int error, const char* msg){
		fprintf(stderr, "%s\n", msg);
	});

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}
	
	m_window = glfwCreateWindow(m_width, m_height, "CIS 565 Path Tracer", NULL, NULL);
	if (!m_window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(m_window);
	
	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	printf("Opengl Version:%s\n", glGetString(GL_VERSION));
	
	//Set up ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	auto io = &ImGui::GetIO(); (void)io;
	ImGui::StyleColorsLight();
	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init("#version 120");

	initCallbacks();
	initCuda();
	initOutputTexture();
	initVAO();
	initPBO();
	
	GLuint passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	m_tracer->initDataContainer(m_guiData.get());

	return true;
}

void Application::initOutputTexture()
{
    glGenTextures(1, &m_imgId);
    glBindTexture(GL_TEXTURE_2D, m_imgId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void Application::initPBO()
{
	// set up vertex data parameter
	int num_texels = m_width * m_height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &m_pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	{
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			fprintf(stderr, "Cuda error: %s: %s.\n", "before", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
	cudaGLRegisterBufferObject(m_pbo);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", "register", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//cudaGraphicsGLRegisterBuffer()
}

void Application::initCuda()
{
	cudaSetDevice(0);

	// Clean up on program exit
	atexit([] {
		Application& app = Application::getInstance();
		app.cleanupGL();
	});
}

void Application::initVAO()
{
	GLfloat vertices[] = {
	-1.0f, -1.0f,
	1.0f, -1.0f,
	1.0f,  1.0f,
	-1.0f,  1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	glGenBuffers(3, m_vertBuff);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertBuff[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(VS_POS_HANDLE, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VS_POS_HANDLE);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertBuff[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer(VS_UV_HANDLE, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VS_UV_HANDLE);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertBuff[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

void Application::initCallbacks()
{
	glfwSetKeyCallback(m_window,[](GLFWwindow * window, int key, int scancode, int action, int mods) {
		Application& app = Application::getInstance();
		if (action == GLFW_PRESS) {
			switch (key) {
			case GLFW_KEY_ESCAPE:
				app.saveImage(app.getOutputImageName().c_str());
				glfwSetWindowShouldClose(window, GL_TRUE);
				break;
			case GLFW_KEY_S:
				app.saveImage(app.getOutputImageName().c_str());
				break;
			case GLFW_KEY_SPACE:
				Camera& cam = app.getCamera();
				cam.lookAt = ogLookAt;
				app.updateCameraView();
				break;
			}
		}
	});
	glfwSetCursorPosCallback(m_window, [](GLFWwindow * window, double xpos, double ypos) {
		Application& app = Application::getInstance();
		if (xpos == app.m_input.lastX || ypos == app.m_input.lastY) return; // otherwise, clicking back into window causes re-start
		if (app.m_input.leftMousePressed) {
			// compute new camera parameters
			phi -= (xpos - app.m_input.lastX) / app.m_width;
			theta -= (ypos - app.m_input.lastY) / app.m_height;
			theta = std::fmax(0.001f, std::fmin(theta, PI));
			app.updateCameraView();
		}
		else if (app.m_input.rightMousePressed) {
			zoom += (ypos - app.m_input.lastY) / app.m_height;
			zoom = std::fmax(0.1f, zoom);
			app.updateCameraView();
		}
		else if (app.m_input.middleMousePressed) {
			Camera& cam = app.getCamera();
			glm::vec3 forward = cam.view;
			forward.y = 0.0f;
			forward = glm::normalize(forward);
			glm::vec3 right = cam.right;
			right.y = 0.0f;
			right = glm::normalize(right);

			cam.lookAt -= (float)(xpos - app.m_input.lastX) * right * 0.01f;
			cam.lookAt += (float)(ypos - app.m_input.lastY) * forward * 0.01f;
			app.updateCameraView();
		}
		app.m_input.lastX = xpos;
		app.m_input.lastY = ypos;
	});
	glfwSetMouseButtonCallback(m_window, [](GLFWwindow * window, int button, int action, int mods) {
		Application& app = Application::getInstance();
		if (app.m_input.mouseOverGuiWindow)
		{
			return;
		}
		app.m_input.leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
		app.m_input.rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
		app.m_input.middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
	});
}

void Application::initCamera()
{
	Camera& cam = getCamera();

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);
	updateCameraView();
}

void Application::resizeWindow(int width, int height)
{
	if (m_window == nullptr) {
		std::cout << "window uninitialized" << std::endl;
	}
	std::cout << "resize window" << std::endl;
	m_width = width;
	m_height = height;
	glfwSetWindowSize(m_window, m_width, m_height);
	glViewport(0, 0, m_width, m_height);
	cleanupGL();
	initOutputTexture();
	initVAO();
	initPBO();
}

GLuint Application::initShader()
{
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

void Application::deletePBO(GLuint* pbo)
{
	if (*pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			fprintf(stderr, "Cuda error: %s: %s.\n", "unregister", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void Application::cleanupGL()
{
	if (m_pbo) {
		deletePBO(&m_pbo);
	}
	if (m_vertBuff[0]) {
		glDeleteBuffers(3, m_vertBuff);
		m_vertBuff[0] = (GLuint)NULL;
		m_vertBuff[1] = (GLuint)NULL;
		m_vertBuff[2] = (GLuint)NULL;
	}
	if (m_imgId) {
		glDeleteTextures(1, &m_imgId);
		m_imgId = (GLuint)NULL;
	}
}

void Application::renderImGui()
{
	auto io = &ImGui::GetIO(); (void)io;
	m_input.mouseOverGuiWindow = io->WantCaptureMouse;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	static float f = 0.0f;
	static int counter = 0;

	ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.

	// LOOK: Un-Comment to check the output window and usage
	//ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
	//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
	//ImGui::Checkbox("Another Window", &show_another_window);

	//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
	//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

	//if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
	//	counter++;
	//ImGui::SameLine();
	//ImGui::Text("counter = %d", counter);
	//ImGui::Text("Traced Depth %d", m_guiData->TracedDepth);
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::End();


	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

std::string Application::currentTimeString()
{
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

glm::vec3 Application::getPixelColor(int x, int y)
{
	int idx = x + (y * m_width);
	glm::vec3 pix = m_scene->state.image[idx];
	return pix;
}

void Application::saveImage(const char* filename)
{
	// output image file
	Image img(m_width, m_height);

	for (int x = 0; x < m_width; x++) {
		for (int y = 0; y < m_height; y++) {
			glm::vec3 pix = getPixelColor(x, y);
			img.setPixel(m_width - 1 - x, y, glm::vec3(pix));
		}
	}
	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

std::string Application::getOutputImageName()
{
	std::string filename = m_scene.get()->state.imageName;
	std::ostringstream ss;
	ss<< ROOT_PATH << "/img/" << filename << "." << currentTimeString() << "." << m_iteration << "samp";
	filename = ss.str();
	return filename;
}

void Application::pathTrace()
{
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (m_iteration == 0) {
		m_tracer.get()->pathtraceInit(m_scene.get());
	}

	if (m_iteration < m_scene->state.maxIterations) {
		uchar4* pbo_dptr = NULL;
		m_iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, m_pbo);

		// execute the kernel
		int frame = 0;
		m_tracer.get()->pathtrace(pbo_dptr, frame, m_iteration);

		// unmap buffer object
		cudaGLUnmapBufferObject(m_pbo);
	}
	else {
		saveImage(getOutputImageName().c_str());
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

Camera& Application::getCamera()
{
	return m_scene.get()->state.camera;
}

void Application::updateCameraView()
{
	m_iteration = 0;
	Camera& cam = getCamera();
	glm::vec3 cameraPosition = glm::vec3(
		zoom * sin(phi) * sin(theta),
		zoom * cos(theta),
		zoom * cos(phi) * sin(theta));

	cam.view = -glm::normalize(cameraPosition);
	glm::vec3 v = cam.view;
	glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
	glm::vec3 r = glm::cross(v, u);
	cam.up = glm::cross(r, v);
	cam.right = r;

	cam.position = cameraPosition;
	cameraPosition += cam.lookAt;
	cam.position = cameraPosition;
}

Application& Application::getInstance()
{
	static Application instance;
	return instance;
}

Application::~Application()
{
	cleanupGL();
}

void Application::loadScene(const char* path)
{
	m_scene = make_unique<Scene>(path);
	m_iteration = 0;
	Camera& cam = getCamera();
	initCamera();
	resizeWindow(cam.resolution.x, cam.resolution.y);
}

void Application::run()
{
	while (!glfwWindowShouldClose(m_window)) {

		glfwPollEvents();

		if(m_scene.get()!=nullptr)pathTrace();

		string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(m_iteration) + " Iterations";
		glfwSetWindowTitle(m_window, title.c_str());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
		glBindTexture(GL_TEXTURE_2D, m_imgId);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// Binding GL_PIXEL_UNPACK_BUFFER back to default
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

		// Render ImGui Stuff
		renderImGui();

		glfwSwapBuffers(m_window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_window);
	glfwTerminate();
}
