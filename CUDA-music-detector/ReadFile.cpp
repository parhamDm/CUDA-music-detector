#include "ReadFile.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include "dirent.h"
#include "DifferenceKernel.h"
using namespace std;

ReadFile::ReadFile()
{
	this->musics = new vector<string>();
	this->musicsList = new vector<vector<float>*>();
	this->samples = new vector<string>();
}

vector<float>* ReadFile::read(string path)
{
	ifstream file;
	ReadArray* ra;
	cout << "reading file from: " << path << endl;
	file.open(path);
	vector<float>* arrayList = new vector<float>();
	string word;
	if (!file)
	{
		cout << "error reading file\n";
		return {};
	}
	while (file >> word)
	{
		arrayList->push_back(stof(word));
	}
	//float * array = (float*)malloc(arrayList.size() * sizeof(float));
	//copy(arrayList.begin(), arrayList.end(), array);
	//ra->array = array;
	//ra->size = arrayList.size();

	return arrayList;
}

bool ReadFile::readMusicDir(string musicDir)
{

	this->musicDir = musicDir;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir((this->musicDir).c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_type == 32768) {
				this->musics->push_back(musicDir + "\\" + ent->d_name);
				printf("File %s found\n", ent->d_name);
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return false;
	}
	return true;
}

bool ReadFile::readSampleDir(string sampleDir)
{
	this->sampleDir = sampleDir;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir((this->sampleDir).c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			if (ent->d_type == 32768) {
				this->samples->push_back(sampleDir + "\\" + (ent->d_name));
				printf("File %s found\n", ent->d_name);
			}
		}
		closedir(dir);
		return true;
	}
	else {
		/* could not open directory */
		perror("");
		return false;
	}

}

bool ReadFile::readMusics()
{
	for (int i = 0; i < musics->size(); i++) {
		vector<float>* v = this->read(musics->at(i));
		musicsList->push_back(v);
	}
	cout << "reading directory completed";
	return true;
}

void ReadFile::testSamples() {
	DifferenceKernel* dk = new DifferenceKernel();
	float answer = 0.0;
	string bestMatch;
	for (int i = 0; i < this->samples->size(); i++) {
		//read sample
		cout << "sample " << samples->at(i) << endl << endl;
		vector<float>* sample = this->read(samples->at(i));

		for (int j = 0; j < this->musicsList->size(); j++) {
			//cout << "music " << musics->at(j) << endl;

			float result = dk->distance(musicsList->at(j), sample);
			if (result > answer) {
				bestMatch = musics->at(j);
				answer = result;
			}
		}
		if (answer < 0.80) {
			cout << "no match found for " << samples->at(i) << endl << endl;
		}
		else {
			cout << "bestmatch for sample " << samples->at(i) << " is " << bestMatch << " " << answer << endl << endl;
		}

		answer = 0.0;
		bestMatch = "";
	}
}
