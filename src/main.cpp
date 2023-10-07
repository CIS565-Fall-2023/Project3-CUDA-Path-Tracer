#include "application.h"
#include "sandbox.h"
#include "pathtrace.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) 
{
	if (argc < 2)
	{
		printf("Please specify a .json scene file to open\n");
		return 1;
	}
	uPtr<SandBox> sandbox = mkU<SandBox>(argv[1]);
	Application app(sandbox->GetCameraResolution());
	app.SetSandBox(sandbox.get());

	sandbox->Init();
	sandbox->m_PathTracer->RegisterPBO(app.pbo);
	app.Run();
	
	sandbox.release();
	
	return 0;
}