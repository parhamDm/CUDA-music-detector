#include "FourierTransform.h"
#include "cufftw.h"
#include "cuda_runtime.h"
#include "stdio.h"
#include "stdlib.h"

#define NX 256
#define BATCH 1
cufftComplex* FourierTransform::transform(float* array, size_t arraySize)
{
	cufftHandle plan;
	cufftComplex *idata;
	cufftComplex *odata;
	cufftComplex *input = (cufftComplex*)malloc(sizeof(cufftComplex)*arraySize);
	//idata = new cufftComplex[arraySize];
#pragma omp parallel for num_threads(8)
	for (int i = 0; i < arraySize; i++) {
		input[i].x = array[i];
		input[i].y = 0;
		//printf("%f ", input[i].x);
		//printf("%f\n", input[i].y);
		//printf("%f\n", idata[i].x);

	}

	cudaMalloc((void**)&odata, sizeof(cufftComplex)*(arraySize + 1)*BATCH);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return NULL;
	}

	cudaMalloc((void**)&idata, sizeof(cufftComplex)*arraySize *BATCH);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return NULL;
	}

	if (cudaMemcpy(idata, input, sizeof(cufftComplex) *arraySize*BATCH, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return NULL;
	}

	if (cufftPlan1d(&plan, arraySize * 2, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return NULL;
	}
	/* Use the CUFFT plan to transform the signal in place. */
	if (cufftExecR2C(plan, (cufftReal*)idata, odata) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		return NULL;
	}

	cufftComplex *result = (cufftComplex *)malloc(sizeof(cufftComplex)*(arraySize / 2 + 1)*BATCH);
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(result, odata, sizeof(cufftComplex)*(arraySize / 2 + 1)*BATCH, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return NULL;
		}
	}
	//	for (long long i = 0; i < 100; i++) {
	//		printf("%.10f ", result[i].x);
	//		printf("%.10f\n", result[i].y);
	//	}

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		return NULL;
	}
	//delete[] idata;
	free(input);
	cufftDestroy(plan);
	//	cudaFree();
	cudaFree(idata);
	cudaFree(odata);
	return result;
}