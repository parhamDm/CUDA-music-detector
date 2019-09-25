#pragma once
#include <string>
#include<vector>

using namespace std;
struct ReadArray {
	float* array;
	size_t size;
};

class ReadFile
{
private:
	vector<string>* musics;
	vector<string>* samples;
	vector<vector<float>*>* musicsList;
	string musicDir;
	string sampleDir;
public:
	ReadFile();
	vector<float>* read(string path);
	bool readMusicDir(string musicDir);
	bool readSampleDir(string sampleDir);
	bool readMusics();
	void testSamples();
};

#pragma once
