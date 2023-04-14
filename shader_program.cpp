#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "shader_program.h"
#include <glad/glad.h>
#include <fstream>
#include <sstream>
#include <iostream>

void ShaderProgram::readCodeFromFiles(const char* vertexShaderPath, const char* fragmentShaderPath,
		char** vertexShaderCode, char** fragmentShaderCode) const {
	std::string vertexShaderCodeStr;
	std::string fragmentShaderCodeStr;
	
	std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try {
		std::stringstream stream;
		file.open(vertexShaderPath);
		stream << file.rdbuf();
		file.close();
		vertexShaderCodeStr = stream.str();
	} catch(std::ifstream::failure) {
		std::cerr << "Error reading file:" << std::endl << vertexShaderPath << std::endl << std::endl;
	}

	try {
		std::stringstream stream;
		file.open(fragmentShaderPath);
		stream << file.rdbuf();
		file.close();
		fragmentShaderCodeStr = stream.str();
	} catch(std::ifstream::failure) {
		std::cerr << "Error reading file:" << std::endl << fragmentShaderPath << std::endl << std::endl;
	}

	const char* tmp = vertexShaderCodeStr.c_str();
	*vertexShaderCode = (char*)malloc(vertexShaderCodeStr.length()*sizeof(char));
	strcpy(*vertexShaderCode, tmp);

	tmp = fragmentShaderCodeStr.c_str();
	*fragmentShaderCode = (char*)malloc(fragmentShaderCodeStr.length()*sizeof(char));
	strcpy(*fragmentShaderCode, tmp);
}

unsigned int ShaderProgram::createShaderProgram(const char* vertexShaderCode,
	const char* fragmentShaderCode) const {
	int success;
	char infoLog[512];

	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderCode, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if(!success) {
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cerr << "Error compiling vertex shader:" << std::endl << infoLog << std::endl;
	}

	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderCode, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if(!success) {
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cerr << "Error compiling fragment shader:" << std::endl << infoLog << std::endl;
	}

	unsigned int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if(!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cerr << "Error linking shader program:" << std::endl << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

ShaderProgram::ShaderProgram(const char* vertexShaderPath, const char* fragmentShaderPath) {
	char* vertexShaderCode;
	char* fragmentShaderCode;

	readCodeFromFiles(vertexShaderPath, fragmentShaderPath, &vertexShaderCode, &fragmentShaderCode);

	ID = createShaderProgram(vertexShaderCode, fragmentShaderCode);
}

void ShaderProgram::use() const {
	glUseProgram(ID);
}

void ShaderProgram::setBoolUniform(const std::string &name, bool value) const {
	glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}

void ShaderProgram::setIntUniform(const std::string &name, int value) const {
	glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void ShaderProgram::setFloatUniform(const std::string &name, float value) const {
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}
