#include "application.h"
#include "sandbox.h"
#include "pathtrace.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) 
{
	uPtr<SandBox> sandbox = mkU<SandBox>(argv[0]);
	Application app(sandbox->GetCameraResolution());
	app.SetSandBox(sandbox.get());

	sandbox->Init();
	sandbox->m_PathTracer->RegisterPBO(app.pbo);
	app.Run();
	
	sandbox.release();
	
	return 0;
}