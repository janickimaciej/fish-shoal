#ifndef SHADER_H
#define SHADER_H

#include <string>

class ShaderProgram {
	void readCodeFromFiles(const char* vertexShaderPath, const char* fragmentShaderPath,
		char** vertexShaderCode, char** fragmentShaderCode) const;
	unsigned int createShaderProgram(const char* vertexShaderCode, const char* fragmentShaderCode) const;
public:
	unsigned int ID;

	ShaderProgram(const char* vertexShaderPath, const char* fragmentShaderPath);
	void use() const;
	void setBoolUniform(const std::string &name, bool value) const;
	void setIntUniform(const std::string &name, int value) const;
	void setFloatUniform(const std::string &name, float value) const;
};

#endif
