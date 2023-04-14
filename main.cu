#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <iostream>
#include <ctime>
#include "shader_program.h"
#include "cuda.cuh"

const unsigned int windowSize = 950;
bool paused = false;
float velocity = 0.1;

GLFWwindow* initializeWindow(int windowSize);
void initializeFish(Fish* fish, unsigned int indices[]);
void createBuffers(unsigned int* VBO, unsigned int* VAO, cudaGraphicsResource_t* cudaVBO,
	Fish* fish, unsigned int indices[]);
void processInput(GLFWwindow* window);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
float c2f(unsigned char color);
float getRandPos();
float getRandDir();

int main() {
	GLFWwindow* window = initializeWindow(windowSize);
	glfwSetKeyCallback(window, keyCallback);

	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	Fish* initialFish = new Fish();
	unsigned int* indices = new unsigned int[3*fishCount]();

	initializeFish(initialFish, indices);

	int* spaceGridHeads;
	int* spaceGridNodes;
	float* newDir;
	allocCuda(&spaceGridHeads, &spaceGridNodes, &newDir, initialFish);
	
	unsigned int VBO;
	unsigned int VAO;
	cudaGraphicsResource_t cudaVBO;
	Fish* fish;
	createBuffers(&VBO, &VAO, &cudaVBO, initialFish, indices);
	delete initialFish;
	delete[] indices;

	ShaderProgram shaderProgram = ShaderProgram("vertexShader.glsl", "fragmentShader.glsl");
	
	//glfwSwapInterval(0); // turn off vsync
	while(!glfwWindowShouldClose(window)) {
		processInput(window);

		cudaGraphicsMapResources(1, &cudaVBO, 0);
		size_t size;
		cudaGraphicsResourceGetMappedPointer((void**)&fish, &size, cudaVBO);
		runKernels(fish, spaceGridHeads, spaceGridNodes, newDir, velocity);
		cudaGraphicsUnmapResources(1, &cudaVBO, 0);

		if(!paused) {
			glClearColor(c2f(11), c2f(5), c2f(49), 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			shaderProgram.use();
			glBindVertexArray(VAO);
			glDrawElements(GL_TRIANGLES, 3*fishCount, GL_UNSIGNED_INT, 0);
			
			glfwSwapBuffers(window);
		}
		glfwPollEvents();
	}

	freeCuda(spaceGridHeads, spaceGridNodes, newDir);
	glfwTerminate();
	return 0;
}

GLFWwindow* initializeWindow(int windowSize) {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(windowSize, windowSize, "Fish", nullptr, nullptr);
	glfwSetWindowPos(window, 0, 38);
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window,
		[](GLFWwindow* window, int width, int height){ glViewport(0, 0, width, height); });
	return window;
}

void initializeFish(Fish* fish, unsigned int indices[]) {
	srand((unsigned int)time(nullptr));
	for(int i = 0; i < fishCount; i++) {
		fish->dir[i] = getRandDir();
		fish->x[i] = getRandPos();
		fish->y[i] = getRandPos();
	}

	for(int i = 0; i < fishCount; i++) {
		for(int j = 0; j < 3; j++) {
			indices[3*i + j] = j*fishCount + i;
		}
	}
}

void createBuffers(unsigned int* VBO, unsigned int* VAO, cudaGraphicsResource_t* cudaVBO,
	Fish* fish, unsigned int indices[]) {
	glGenBuffers(1, VBO);
	unsigned int EBO;
	glGenBuffers(1, &EBO);
	glGenVertexArrays(1, VAO);
	glBindVertexArray(*VAO);
	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Fish), fish, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(cudaVBO, *VBO, cudaGraphicsMapFlagsNone);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3*fishCount*sizeof(unsigned int), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(float),
		(void*)(fishCount*sizeof(float)));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float),
		(void*)(4*fishCount*sizeof(float)));
	glEnableVertexAttribArray(1);
}

void processInput(GLFWwindow* window) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	static const float velocityMultiplier = 1.2;
    if(key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		paused = !paused;
    }
	if(key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		if(velocity > 0.001) velocity /= velocityMultiplier;
	}
	if(key == GLFW_KEY_UP && action == GLFW_PRESS) {
		if(velocity < 0.25) velocity *= velocityMultiplier;
	}
}

float getRandPos() {
	return (float)rand()/RAND_MAX*2 - 1;
}

float getRandDir() {
	return (float)rand()/RAND_MAX*2*Pi;
}

float c2f(unsigned char color) {
	if(color < 0) return 0;
	if(color > 255) return 1;
	return (float)color/255;
}
