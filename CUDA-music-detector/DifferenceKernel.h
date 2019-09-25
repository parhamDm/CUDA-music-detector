#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufftw.h"
#include <vector>
using namespace std;
class DifferenceKernel
{
public:
	float distance(vector<float> musicVector, vector<float> sample);
	float distance(vector<float>* musicVector, vector<float>* sampleVector);
};

