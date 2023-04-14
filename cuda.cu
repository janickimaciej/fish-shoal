#include "cuda.cuh"
#include <iostream>
#include <chrono>
#include <cmath>

constexpr float Eps = 1e-5;
constexpr float range = 0.1;
constexpr float legLength = 0.02;
constexpr float legAngleDeg = 15;
constexpr float legAngleRad = legAngleDeg/180*Pi;
constexpr float FOVDeg = 300;
constexpr float FOVRad = FOVDeg/180*Pi;
const int spaceGridDim = ceil(2.0/range - Eps);

void allocCuda(int** spaceGridHeads, int** spaceGridNodes, float** newDir, Fish* initialFish) {
	cudaMalloc(spaceGridHeads, spaceGridDim*spaceGridDim*sizeof(int));
	cudaMalloc(spaceGridNodes, fishCount*sizeof(int));
	cudaMalloc(newDir, fishCount*sizeof(float));
	cudaMemcpy(*newDir, initialFish->dir, fishCount*sizeof(float), cudaMemcpyHostToDevice);
}

void freeCuda(int* spaceGridHeads, int* spaceGridNodes, float* newDir) {
	cudaFree(spaceGridHeads);
	cudaFree(spaceGridNodes);
	cudaFree(newDir);
}

void runKernels(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, float* newDir, float velocity) {
	initializeSpaceGrid(spaceGridHeads);
	fillSpaceGrid(fish, spaceGridHeads, spaceGridNodes);
	fishInteraction(fish, spaceGridHeads, spaceGridNodes, newDir);

	static std::chrono::steady_clock::time_point prevTime = std::chrono::high_resolution_clock::now();
	std::chrono::steady_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
	float deltaTime =
		std::chrono::duration_cast<std::chrono::microseconds>(currentTime - prevTime).count()/1e6;
	prevTime = currentTime;

	static unsigned int counter = 0; // fps
	counter++;
	if(!(counter%60)) {
		std::cout << 1/deltaTime << std::endl;
	}
	
	fishMove(fish, deltaTime, newDir, velocity);
	cudaDeviceSynchronize();
}

void initializeSpaceGrid(int* spaceGridHeads) {
	static const int blockSize = 256;
	static const int numBlocks = (spaceGridDim*spaceGridDim + blockSize - 1)/blockSize;
	initializeSpaceGridKernel<<<numBlocks, blockSize>>>(spaceGridHeads, spaceGridDim*spaceGridDim);
}

void fillSpaceGrid(Fish* fish, int* spaceGridHeads, int* spaceGridNodes) {
	static const int blockSize = 256;
	static const int numBlocks = (fishCount + blockSize - 1)/blockSize;
	fillSpaceGridKernel<<<numBlocks, blockSize>>>(fish, spaceGridHeads, spaceGridNodes, spaceGridDim);
}

void fishInteraction(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, float* newDir) {
	static const int blockSize = 256;
	static const int numBlocks = (fishCount + blockSize - 1)/blockSize;
	fishInteractionKernel<<<numBlocks, blockSize>>>(fish, spaceGridHeads, spaceGridNodes, spaceGridDim,
		newDir);
}

void fishMove(Fish* fish, float deltaTime, float* newDir, float velocity) {
	static const int blockSize = 256;
	static const int numBlocks = (fishCount + blockSize - 1)/blockSize;
	fishMoveKernel<<<numBlocks, blockSize>>>(fish, velocity, deltaTime, newDir);
}

__global__ void initializeSpaceGridKernel(int* spaceGridHeads, int spaceGridCount) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= spaceGridCount) return;

	spaceGridHeads[tid] = -1;
}

__global__ void fillSpaceGridKernel(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, int spaceGridDim) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= fishCount) return;

	int spaceGridCellX = (fish->x[tid] + 1)/range;
	int spaceGridCellY = (fish->y[tid] + 1)/range;
	int spaceGridCell = spaceGridCellY*spaceGridDim + spaceGridCellX;

	int nextNode = atomicExch(spaceGridHeads + spaceGridCell, tid);
	spaceGridNodes[tid] = nextNode;
}

__global__ void fishInteractionKernel(Fish* fish, int* spaceGridHeads, int* spaceGridNodes, int spaceGridDim,
	float* newDir) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= fishCount) return;
	
	if(wallAvoidance(tid, fish, newDir)) return;

	int firstSpaceGridCellX = (fish->x[tid] + 1)/range - 0.5;
	int firstSpaceGridCellY = (fish->y[tid] + 1)/range - 0.5;

	if(firstSpaceGridCellX >= 0 && firstSpaceGridCellY >= 0) {
		spaceGridCellInteraction(tid, firstSpaceGridCellY*spaceGridDim + firstSpaceGridCellX,
			fish, spaceGridHeads, spaceGridNodes, newDir);
	}
	if(firstSpaceGridCellX + 1 < spaceGridDim && firstSpaceGridCellY >= 0) {
		spaceGridCellInteraction(tid, firstSpaceGridCellY*spaceGridDim + firstSpaceGridCellX + 1,
			fish, spaceGridHeads, spaceGridNodes, newDir);
	}
	if(firstSpaceGridCellX >= 0 && firstSpaceGridCellY + 1 < spaceGridDim) {
		spaceGridCellInteraction(tid, (firstSpaceGridCellY + 1)*spaceGridDim + firstSpaceGridCellX,
			fish, spaceGridHeads, spaceGridNodes, newDir);
	}
	if(firstSpaceGridCellX + 1 < spaceGridDim && firstSpaceGridCellY + 1 < spaceGridDim) {
		spaceGridCellInteraction(tid, (firstSpaceGridCellY + 1)*spaceGridDim + firstSpaceGridCellX + 1,
			fish, spaceGridHeads, spaceGridNodes, newDir);
	}
}

__device__ bool wallAvoidance(int tid, Fish* fish, float* newDir) {
	const float increment = 0.05;
	float marginDeg = 30 + tid%41 - 20;
	float marginRad = marginDeg/180*Pi;

	float x = fish->x[tid];
	float y = fish->y[tid];
	float dir = getNormalizedAngle(fish->dir[tid]);

	if(x < -1 + range && (dir > Pi/2 - marginRad && dir < 3*Pi/2 + marginRad)) { // left
		newDir[tid] += dir > Pi ? increment : -increment;
		return true;
	}
	if(x > 1 - range && (dir > 3*Pi/2 - marginRad || dir < Pi/2 + marginRad)) { // right
		newDir[tid] += dir < Pi ? increment : -increment;
		return true;
	}
	if(y < -1 + range && (dir > Pi - marginRad || dir < marginRad)) { // bottom
		newDir[tid] += (dir < Pi/2 || dir > 3*Pi/2) ? increment : -increment;
		return true;
	}
	if(y > 1 - range && (dir > 2*Pi - marginRad || dir < Pi + marginRad)) { // top
		newDir[tid] += (dir > Pi/2 && dir < 3*Pi/2) ? increment : -increment;
		return true;
	}
	return false;
}

__device__ void spaceGridCellInteraction(int tid, int spaceGridCell, Fish* fish, int* spaceGridHeads,
	int* spaceGridNodes, float* newDir) {
	float tidX = fish->x[tid];
	float tidY = fish->y[tid];
	float tidDir = newDir[tid];

	const float separationWeight = 0.01;
	const float alignmentWeight = 0.01;
	const float cohesionWeight = 0.01;
	const float currentWeight = 1 - separationWeight - alignmentWeight - cohesionWeight;

	int counter = 0;
	float separationX = 0;
	float separationY = 0;
	float alignmentX = 0;
	float alignmentY = 0;
	float averagePosX = 0;
	float averagePosY = 0;

	int node = spaceGridHeads[spaceGridCell];
	while(node != -1) {
		if(areInteracting(fish, tid, node)) {
			float nodeX = fish->x[node];
			float nodeY = fish->y[node];
			float nodeDir = fish->dir[node];
			float distance = sqrt((nodeX - tidX)*(nodeX - tidX) + (nodeY - tidY)*(nodeY - tidY));

			separationX += (tidX - nodeX)/distance/distance;
			separationY += (tidY - nodeY)/distance/distance;
			averagePosX += nodeX;
			averagePosY += nodeY;
			alignmentX += cos(nodeDir);
			alignmentY += sin(nodeDir);
			counter++;
		}
		node = spaceGridNodes[node];
	}
	if(counter == 0) return;

	normalizeVector(&separationX, &separationY);

	normalizeVector(&alignmentX, &alignmentY);

	averagePosX /= counter;
	averagePosY /= counter;
	float cohesionX = averagePosX - tidX;
	float cohesionY = averagePosY - tidY;
	normalizeVector(&cohesionX, &cohesionY);

	float currentX = cos(tidDir);
	float currentY = sin(tidDir);
	normalizeVector(&currentX, &currentY);

	float newX = currentWeight*currentX + separationWeight*separationX + alignmentWeight*alignmentX +
		cohesionWeight*cohesionX;
	float newY = currentWeight*currentY + separationWeight*separationY + alignmentWeight*alignmentY +
		cohesionWeight*cohesionY;

	newDir[tid] = atan2(newY, newX);
}

__device__ bool areInteracting(Fish* fish, int receiver, int sender) {
	if(receiver == sender) return false;

	float dX = fish->x[sender] - fish->x[receiver];
	float dY = fish->y[sender] - fish->y[receiver];
	if(dX*dX + dY*dY > range*range) return false;

	float relativeAngle = atan2(dY, dX) - fish->dir[receiver];
	relativeAngle = getNormalizedAngle(relativeAngle);
	if(relativeAngle > FOVRad/2 && relativeAngle < 2*Pi - FOVRad) return false;
	
	return true;
}

__device__ float getNormalizedAngle(float angle) {
	while(angle < 0) angle += 2*Pi;
	while(angle >= 2*Pi) angle -= 2*Pi;
	return angle;
}

__device__ void normalizeVector(float* x, float* y) {
	float norm = sqrt((*x)*(*x) + (*y)*(*y));
	*x /= norm;
	*y /= norm;
}

__global__ void fishMoveKernel(Fish* fish, float velocity, float deltaTime, float* newDir) {
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= fishCount) return;

	float x = fish->x[tid];
	float y = fish->y[tid];
	float dir = newDir[tid];

	x = x + cos(dir)*velocity*deltaTime;
	y = y + sin(dir)*velocity*deltaTime;

	if(x < -1) x = -1;
	if(y < -1) y = -1;
	if(x > 1) x = 1;
	if(y > 1) y = 1;

	fish->x[tid] = x;
	fish->y[tid] = y;
	fish->dir[tid] = dir;

	float basicAngle = dir + Pi;
	fish->lx[tid] = x + legLength*cos(basicAngle - legAngleRad);
	fish->ly[tid] = y + legLength*sin(basicAngle - legAngleRad);
	fish->rx[tid] = x + legLength*cos(basicAngle + legAngleRad);
	fish->ry[tid] = y + legLength*sin(basicAngle + legAngleRad);
}
