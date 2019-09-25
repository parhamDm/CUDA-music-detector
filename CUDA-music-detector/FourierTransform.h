#pragma once
#include"cufft.h"
class FourierTransform
{
public:
	cufftComplex * transform(float * array, cufftComplex * result, size_t arraySize);
	cufftComplex * transform(float * array, size_t arraySize);
};