#ifndef CUDA_CUH
#define CUDA_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr float Pi = 3.14159265358979323846;
const unsigned int fishCount = 1e4;

struct Fish {
	float dir[fishCount];
	float x[fishCount];
	float lx[fishCount];
	float rx[fishCount];
	float y[fishCount];
	float ly[fishCount];
	float ry[fishCount];
};

void allocCuda(int** spaceGridHeads, int** spaceGridNodes, float** newDir, Fish* initialFish);
void freeCuda(int* spaceGridHeads, int* spaceGridNodes, float* newDir);
void runKernels(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, float* newDir, float velocity);
void initializeSpaceGrid(int* spaceGridHeads);
void fillSpaceGrid(Fish* fish, int* spaceGridHeads, int* spaceGridNodes);
void fishInteraction(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, float* newDir);
void fishMove(Fish* fish, float deltaTime, float* newDir, float velocity);
__global__ void initializeSpaceGridKernel(int* spaceGridHeads, int spaceGridCount);
__global__ void fillSpaceGridKernel(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, int spaceGridDim);
__global__ void fishInteractionKernel(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, int spaceGridDim,
	float* newDir);
__device__ bool wallAvoidance(int tid, Fish* fish, float* newDir);
__device__ void spaceGridCellInteraction(int tid, int spaceGridCell, Fish* fish, int* spaceGridHeads,
	int* spaceGridNodes, float* newDir);
__device__ bool areInteracting(Fish* fish, int receiver, int sender);
__device__ float getNormalizedAngle(float angle);
__device__ void normalizeVector(float* x, float* y);
__global__ void fishMoveKernel(Fish* fish, float velocity, float deltaTime, float* newDir);

#endif
