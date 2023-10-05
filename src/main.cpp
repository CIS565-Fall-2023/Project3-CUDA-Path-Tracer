#include "application.h"
#include "sandbox.h"
#include "pathtrace.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) 
{
	Application app({1280, 960});
	SandBox sandbox;
	app.SetSandBox(&sandbox);
	
	sandbox.m_PathTracer->RegisterPBO(app.pbo);
	app.Run();
	return 0;
}