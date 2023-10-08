#pragma once
#include <ctime>
//#include "main.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
#include "pathtrace.h"
#include <GL/glew.h>
#include <memory>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "glslUtility.hpp"
#include "pathtrace.h"
#include "image.h"
struct UserInput {
    bool mouseOverGuiWindow;
    // For camera controls
    bool leftMousePressed = false;
    bool rightMousePressed = false;
    bool middleMousePressed = false;
    double lastX;
    double lastY;
};

class Application
{
public:
    int m_width;
    int m_height;
    int m_iteration;
private:
    GLFWwindow* m_window;
    std::unique_ptr<Scene> m_scene;
    std::unique_ptr<PathTracer> m_tracer;
    std::unique_ptr<GuiDataContainer> m_guiData;
    
    GLuint m_imgId;
    GLuint m_pbo = (GLuint)NULL;
    GLuint m_vertBuff[3] = { (GLuint)NULL,(GLuint)NULL ,(GLuint)NULL };
    
    Application();
    Application(const Application& _ref);
    Application& operator=(const Application& ref);
    
    bool init();
    void initOutputTexture();
    void initPBO();
    void initCuda();
    void initVAO();
    void initCallbacks();
    void initCamera();
    void resizeWindow(int width,int height);
    GLuint initShader();
    
    void deletePBO(GLuint* pbo);
    
    void renderImGui();
    std::string currentTimeString();
    glm::vec3 getPixelColor(int x, int y);
    
    void pathTrace();
    
public:
    UserInput m_input;
    static Application& getInstance();
    ~Application();
    void loadScene(const char* path);
    void saveImage(const char* filename);
    std::string getOutputImageName();
    void cleanupGL();
    void run();

    Camera& getCamera();
    void updateCameraView();
};