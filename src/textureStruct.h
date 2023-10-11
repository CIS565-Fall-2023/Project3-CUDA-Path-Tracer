#pragma once
#include "glm/glm.hpp"
#include <vector>

enum TextureType {
	TEXTURE_TYPE_DIFFUSE,
	TEXTURE_TYPE_SPECULAR,
	TEXTURE_TYPE_NORMAL,
	TEXTURE_TYPE_HEIGHT
};

struct TextureInfo {
	int width;
	int height;
	int nrChannels;
	std::vector<unsigned char> data;
};

struct Texture {
	int width;
	int height;
	int nrChannels;
	unsigned char* data;
};

class TextureManager {
private:
	static TextureManager* instance;
	TextureManager();
public:
	static TextureManager* getInstance();
	Texture loadTexture(const char* path, TextureType type);
	void bindTexture(Texture texture);
	void unbindTexture();
	void deleteTexture(Texture texture);
};