#include "application.h"

#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <functional>

#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include "common.h"
#include <cuda_gl_interop.h>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include "utilities.h"

#include "sandbox.h"

Application* Application::s_Instance = nullptr;

// callback function that used to invoke application's callback function
// since member functions cannot be used as callback directly

void WindowResizeCallback(GLFWwindow* window, int w, int h)
{
	Application::GetApplication()->OnWindowResize(window, w, h);
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	Application::GetApplication()->OnMouseButton(window, button, action, mods);
}

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	Application::GetApplication()->OnKey(window, key, scancode, action, mods);
}

void MousePositionCallback(GLFWwindow* window, double x, double y)
{
	Application::GetApplication()->OnMousePosition(window, x, y);
}

void ErrorCallback(int error, const char* description) 
{
	fprintf(stderr, "%s\n", description);
}

Application::Application(const glm::ivec2& win_size)
	:m_Resolution(win_size)
{
	assert(!s_Instance);
	s_Instance = this;
	Init();
}

Application::~Application()
{
	glDeleteTextures(1, &displayImage);
	glDeleteBuffers(1, &pbo);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_GLFWwindow);
	glfwTerminate();

	s_Instance = nullptr;
}

void Application::InitVAO()
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

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

void Application::InitShader()
{
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	glUseProgram(program);
}

void Application::ResizeGL()
{
	// resize texture
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_Resolution.x, m_Resolution.y, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// resize PBO
	// set up vertex data parameter
	int num_texels = m_Resolution.x * m_Resolution.y;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
}

bool Application::Init()
{
	glfwSetErrorCallback(ErrorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	m_GLFWwindow = glfwCreateWindow(m_Resolution.x, m_Resolution.y, "CIS 565 Path Tracer", NULL, NULL);
	if (!m_GLFWwindow) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(m_GLFWwindow);

	glfwSetWindowSizeCallback(m_GLFWwindow, WindowResizeCallback);
	glfwSetKeyCallback(m_GLFWwindow, KeyCallback);
	glfwSetCursorPosCallback(m_GLFWwindow, MousePositionCallback);
	glfwSetMouseButtonCallback(m_GLFWwindow, MouseButtonCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	printf("Opengl Version:%s\n", glGetString(GL_VERSION));
	//Set up ImGui

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); (void)io;
	ImGui::StyleColorsLight();
	ImGui_ImplGlfw_InitForOpenGL(m_GLFWwindow, true);
	ImGui_ImplOpenGL3_Init("#version 120");
	
	// Initialize other stuff
	InitVAO();
	glGenTextures(1, &displayImage);
	glGenBuffers(1, &pbo);

	InitShader();
	ResizeGL();

	glActiveTexture(GL_TEXTURE0);

	return true;
}

void Application::OnWindowResize(GLFWwindow* window, int x, int y)
{
	m_Resolution.x = x;
	m_Resolution.y = y;

	ResizeGL();
	m_SandBox->OnWindowResize(x, y);
}

void Application::OnMouseButton(GLFWwindow* window, int button, int action, int mods)
{
	m_SandBox->OnMouseButton(button, action, mods);
}

void Application::OnKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	m_SandBox->OnKey(key, scancode, action, mods);
}

void Application::OnMousePosition(GLFWwindow* window, double x, double y)
{
	m_SandBox->OnMousePosition(x, y);
}

bool Application::Run()
{
	while (!glfwWindowShouldClose(m_GLFWwindow)) 
	{
		glfwPollEvents();

		m_SandBox->Run();

		std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(0) + " Iterations";
		glfwSetWindowTitle(m_GLFWwindow, title.c_str());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Resolution.x, m_Resolution.y, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// Binding GL_PIXEL_UNPACK_BUFFER back to default
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Render ImGui Stuff
		m_SandBox->DrawImGui();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(m_GLFWwindow);
	}
	return 0;
}