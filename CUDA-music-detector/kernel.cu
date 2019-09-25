
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FourierTransform.h"
#include <stdio.h>
#include "ReadFile.h"
#include<iostream>
#include "cufftw.h"
#include "DifferenceKernel.h"
#include "Cuda_Help.cuh"
#include <vector>

//#define BLOCK_DIM 1024
using namespace std;

int main(int argc, char** argv)
{
	FourierTransform ft = FourierTransform();
	ReadFile * rf = new ReadFile();

	string musicDir;
	string sampleDir;

	bool get_from_user = true;
	if (argc == 3) {
		get_from_user = false;
		musicDir = string(argv[1]);
		sampleDir = string(argv[2]);
	}

	while (true)
	{
		cout << "enter MusicDir :" << endl << ">>";
		if (get_from_user) { cin >> musicDir; };
		bool result = rf->readMusicDir(musicDir);
		if (result) { break; }
		else { get_from_user = true; }
	}
	cout << sampleDir << endl;
	while (true)
	{
		cout << "enter sample dir :" << endl << ">>";
		if (get_from_user) { cin >> sampleDir; };
		bool result = rf->readSampleDir(sampleDir);
		if (result) { break; }
		else { get_from_user = true; }
	}

	rf->readMusics();
	rf->testSamples();


	//rf.readMusicDir();
	//rf.readSampleDir();
	//rf.testSamples();
	//return 0;
	//vector<float>* ra = rf.read("D:\\misc\\ffmpg\\Data\\09. Run Like Hell.txt");
	//vector<float>* ra1 = rf.read("D:\\misc\\ffmpg\\Data\\02. Is There Anybody Out There_sample.txt");
	//for (int i = 1000; i < 20000; i++)
	//cufftComplex* cufft1 = ft.transform(ra1->array, ra1->size);
	//cout << endl;
	//cufftComplex* cufft = ft.transform(ra->array, ra->size);
	//DifferenceKernel dk;
	//dk.distance(ra,ra1);
	//float* result=(float *)malloc(2*sizeof(float));
	//result[0] = 0.0;
	//result[0] = 1.0;
	//dist(cufft1, cufft,(ra->size /2)+1,result);

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
