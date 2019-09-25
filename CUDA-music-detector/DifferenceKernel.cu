#include"DifferenceKernel.h"
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufftw.h"
#include "Cuda_Help.cuh"
#include <stdio.h>
#include <iostream>
#include <vector>
#include "FourierTransform.h"
using namespace std;
#define BLOCK_DIM 512

__global__  void distanceKernel(cufftComplex* s1, cufftComplex* s2, float* result, size_t size)
{
	__shared__ float numberx[1024];
	__shared__ float numbery[1024];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;

	//printf("%d\n", result);
	//calculate currentnumber
	numberx[tid] = 0.0;
	numbery[tid] = 0.0;

	if (index < size) {

		numberx[tid] = abs(s1[index].x - s2[index].x);
		numbery[tid] = abs(s1[index].y - s2[index].y);
		//		atomicAdd(&(result[0]), numberx[tid]);
		//		atomicAdd(&(result[1]), numbery[tid]);

		//printf("%f\n", numbery[tid]);
	}
	else {
		//	printf("%d\n", size);
		//printf("%f\n", s2[881903].x);
		numberx[tid] = 0.0;
		numbery[tid] = 0.0;
	}
	//	return;

	__syncthreads();
	//reduction
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s) {
			numberx[tid] += numberx[tid + s];
			numbery[tid] += numbery[tid + s];
		}
		__syncthreads();
	}
	/*if (tid < 32)
	{
	numberx[tid] += numberx[tid + 32];
	numbery[tid] += numbery[tid + 32];
	numberx[tid] += numberx[tid + 16];
	numbery[tid] += numbery[tid + 16];
	numberx[tid] += numberx[tid + 8];
	numbery[tid] += numbery[tid + 8];
	numberx[tid] += numberx[tid + 4];
	numbery[tid] += numbery[tid + 4];
	numberx[tid] += numberx[tid + 2];
	numbery[tid] += numbery[tid + 2];
	numberx[tid] += numberx[tid + 1];
	numbery[tid] += numbery[tid + 1];
	}*/

	if (tid == 0) {
		atomicAdd(&(result[0]), numberx[0]);
		atomicAdd(&(result[1]), numbery[0]);
		//printf("%f %f\n",result[0], result[1]);
	}

}

__global__  void cosineSimilarity(cufftComplex* s1, cufftComplex* s2, float* result, size_t size)
{
	__shared__ float t1[BLOCK_DIM];
	__shared__ float t2[BLOCK_DIM];
	__shared__ float t3[BLOCK_DIM];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	int blockSize = blockDim.x;
	//printf("%d\n", result);
	//calculate currentnumber
	t1[tid] = 0.0;
	t2[tid] = 0.0;
	t3[tid] = 0.0;

	if (index < size) {
		//printf("%d\n", result);

		float x = sqrtf(s1[index].x * s1[index].x + s1[index].y*s1[index].y);
		float y = sqrtf(s2[index].x * s2[index].x + s2[index].y*s2[index].y);
		t1[tid] += x*y;
		t2[tid] += x*x;
		t3[tid] += y*y;
	}
	else {
		t1[tid] = 0.0;
		t2[tid] = 0.0;
		t3[tid] = 0.0;
	}


	__syncthreads();

	//reduction

	if (blockSize >= 1024) {
		if (tid < 512) {
			t1[tid] += t1[tid + 512];
			t2[tid] += t2[tid + 512];
			t3[tid] += t3[tid + 512];
		} __syncthreads();
	}

	if (blockSize >= 512) {
		if (tid < 256) {
			t1[tid] += t1[tid + 256];
			t2[tid] += t2[tid + 256];
			t3[tid] += t3[tid + 256];
		} __syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) {
			t1[tid] += t1[tid + 128];
			t2[tid] += t2[tid + 128];
			t3[tid] += t3[tid + 128];
		} __syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			t1[tid] += t1[tid + 64];
			t2[tid] += t2[tid + 64];
			t3[tid] += t3[tid + 64];
		} __syncthreads();
	}

	if (tid < 32)
	{
		t1[tid] += t1[tid + 32];
		t1[tid] += t1[tid + 16];
		t1[tid] += t1[tid + 8];
		t1[tid] += t1[tid + 4];
		t1[tid] += t1[tid + 2];
		t1[tid] += t1[tid + 1];

		t2[tid] += t2[tid + 32];
		t2[tid] += t2[tid + 16];
		t2[tid] += t2[tid + 8];
		t2[tid] += t2[tid + 4];
		t2[tid] += t2[tid + 2];
		t2[tid] += t2[tid + 1];

		t3[tid] += t3[tid + 32];
		t3[tid] += t3[tid + 16];
		t3[tid] += t3[tid + 8];
		t3[tid] += t3[tid + 4];
		t3[tid] += t3[tid + 2];
		t3[tid] += t3[tid + 1];
	}
	/*
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
	if (tid < s) {
	t1[tid] += t1[tid + s];
	t2[tid] += t2[tid + s];
	t3[tid] += t3[tid + s];
	}
	__syncthreads();
	}
	//	printf("%f\n\n", result[1]);
	*/
	if (tid == 0) {
		atomicAdd(&(result[0]), t1[0]);
		atomicAdd(&(result[1]), t2[0]);
		atomicAdd(&(result[2]), t3[0]);
	}

}

cudaError_t dist(cufftComplex* s1, cufftComplex* s2, size_t size, float * answer)
{
	cudaError_t cudaStatus;
	cufftComplex * dev_s1;
	cufftComplex * dev_s2;
	float * result;
	//cout << s1[0].y << endl;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_s1, size * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 0");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_s2, size * sizeof(cufftComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 1");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&result, 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! 2");
		goto Error;
	}
	//copy data
	cudaStatus = cudaMemcpy(dev_s1, s1, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! 3");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_s2, s2, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! 4");
		goto Error;
	}
	//	cudaStatus = cudaMemcpy(result, answer, 2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed! 5");
		goto Error;
	}

	int blockSize = BLOCK_DIM;
	int gridSize = (size - 1) / 1024 + 1;
	//cout << gridSize<<" "<<size<<endl;
	float* dev_c;
	cudaMalloc((void**)&dev_c, size * sizeof(int));

	//calculate kernel time
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	//distanceKernel<<<gridSize, blockSize >>>(dev_s1, dev_s2, result, size);
	cosineSimilarity << <gridSize, blockSize >> >(dev_s1, dev_s2, result, size);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(answer, result, 3 * sizeof(float), cudaMemcpyDeviceToHost);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy last failed!");
		goto Error;
	}
	//printf("serial ans : %f\n", s2[881903].x);

	float ansx = 0.0;
	float ansy = 0.0;

	//serialValidation
	for (int i = 0; i < size; i++) {
		ansx += abs(s1[i].x - s2[i].x);
		ansy += abs(s1[i].y - s2[i].y);
		//ans += sqrt(pow(x, 2) + pow(y, 2));
	}
	/*
	//serial cosine distance
	float t1=0, t2=0, t3=0;
	for (int i = 0; i < size; i++) {
	float x = sqrtf(s1[i].x * s1[i].x + s1[i].y*s1[i].y);
	float y = sqrtf(s2[i].x * s2[i].x + s2[i].y*s2[i].y);
	t1 += x*y;
	t2 += x*x;
	t3 += y*y;
	//ansy += abs(s1[i].y - s2[i].y);
	//ans += sqrt(pow(x, 2) + pow(y, 2));
	}
	//cout << size<<endl;
	//printf("GPU answer : %f + %f i\n", answer[0]/1000000,answer[1] / 1000000);
	//printf("SERIAL answer : %f + %f i", ansx / 1000000, ansy / 1000000);
	//cout << (answer[0]) <<" "<< answer[1]<<" " << answer[2]<<endl;
	printf("cosineDistance = %f \n", abs(t1) / (sqrtf(t2)*sqrtf(t3)));
	*/
	//todo uncomment for enabling log
	//printf("cosineDistance GPU = %f \n", abs(answer[0]) / (sqrtf(answer[1])*sqrtf(answer[2])));
	//printf("Elapsed time : %f ms\n\n", elapsedTime);

	answer[0] = abs(answer[0]) / (sqrtf(answer[1])*sqrtf(answer[2]));

Error:
	cudaFree(dev_s1);
	cudaFree(dev_s2);
	cudaFree(result);
	return cudaStatus;
}


float DifferenceKernel::distance(vector<float>* musicVector, vector<float>* sampleVector)
{
	int musicSize = musicVector->size();
	int sampleSize = sampleVector->size();
	int steps = musicSize / sampleSize * 2;
	float* ans = (float *)malloc(3 * sizeof(float));
	float maxSimilarity = 0.0;
	FourierTransform fourierTransform;

	float * sample = (float*)malloc(sampleSize * sizeof(float));
	float * music = (float*)malloc(sampleSize * sizeof(float));

	copy(sampleVector->begin(), sampleVector->end(), sample);
	cufftComplex * smpl = fourierTransform.transform(sample, sampleSize);
	cufftComplex * msc;
	for (int i = 0; i < steps; i++) {
		if (i*(sampleSize / 2) + sampleSize >= musicSize) {
			break;
		}
#pragma omp parallel for num_threads(8)
		for (int j = i*(sampleSize / 2); j < i * (sampleSize / 2) + sampleSize; j++) {
			music[j - i*sampleSize / 2] = musicVector->at(j);
			//cout << j << " "<< music[j - i*sampleSize];
		}

		//cout << ((i + 1)*sampleSize) - i*sampleSize << endl;;

		//fourier transform
		msc = fourierTransform.transform(music, sampleSize);

		//lunch kernell
		dist(smpl, msc, sampleSize / 2 + 1, ans);

		if (ans[0] > maxSimilarity) {
			maxSimilarity = ans[0];
		}
		free(msc);
		//msc = NULL;
	}
	//very rare case;
	if (musicSize - sampleSize == 0) {
		return maxSimilarity;
	}
#pragma omp parallel for num_threads(8)
	for (int j = musicSize - sampleSize; j < musicSize; j++) {
		music[j - (musicSize - sampleSize)] = musicVector->at(j);

		//cout << j << " " << music[j - i*musicSize];
	}
	msc = fourierTransform.transform(music, sampleSize);
	dist(smpl, msc, sampleSize / 2 + 1, ans);
	if (ans[0] > maxSimilarity) {
		maxSimilarity = ans[0];
	}

	free(msc);
	free(smpl);
	free(music);
	free(ans);
	cudaDeviceReset();
	//copy(musicSize-sampleSize, musicSize, music);
	//cout << musicSize - sampleSize << " " << musicSize;
	cout << "max similarity found :" << maxSimilarity << endl;
	return maxSimilarity;
}

