#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;
#include "imdb_sdk.h"
#define DEBUG_INFO 1

void LoadImages(const string &strImagePath, const string &strTimesStampsPath,
                vector<string> &strImagesFileNames)
{
    if(DEBUG_INFO) std::cout << "LoadImages(): " << strTimesStampsPath << std::endl;
    ifstream fTimes;
    fTimes.open(strTimesStampsPath.c_str());
    strImagesFileNames.reserve(5000);
    while (!fTimes.eof())
        {
            string s;
            getline(fTimes, s);
            if (!s.empty())
            {
                char c = s.at(0); //skip the first title line
                if (c < '0' || c > '9')
                {
                    getline(fTimes, s);
                    continue;
                }
                stringstream ss;
                ss << s;
                string imgFile = strImagePath + "/" + ss.str() + ".png";
                strImagesFileNames.push_back(imgFile);
                double t;
                ss >> t;
            }
        }
    fTimes.close();
}

void LoadPathList( const string &fileListsPath, vector<string> &fileVec)
{
    fileVec.clear();
    if(DEBUG_INFO) std::cout << "LoadImages(): " << fileListsPath << std::endl;
    ifstream fTimes;
    fTimes.open(fileListsPath.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            if(DEBUG_INFO) std::cout << "load set: " << ss.str() << std::endl;
            fileVec.push_back(ss.str());
        }
    }
    fTimes.close();
}


int main()
{
    int addStep = 10;
    int queryStep = 10;
    string voc_path= "../config/loopC_vocdata.bin";
    string fileListsPath = "../config/datalist_server.txt";
    std::string pattern_file = "../config/loopC_pattern.yml";
//    ImageDatabase imdb(voc_path, pattern_file);
    void* handler = initDataBase(voc_path, pattern_file);
    vector<string> fileVec;
    LoadPathList(fileListsPath, fileVec);
    double loadImageTimeCost, queryImageTimeCost;
    int counterDb = 0, counterQuery = 0;
//    vector<BRIEF::bitset> brief_descriptors;
    for(auto i = 0; i < fileVec.size(); i++) {
        string base_path = fileVec[i];
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            cout << "The size of image list is " << imagesList.size() << endl;
        }
        for (int ni = 0; ni < imagesList.size(); ni += addStep) {
//            cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
//            imdb.extractFeatureVector(image, brief_descriptors);
//            imdb.addImage(image, i);
              std::cout << "load image: " << imagesList[ni] << std::endl;
              addImage(handler, imagesList[ni],i);
        }
        counterDb += imagesList.size() / addStep;
    }


        ifstream myfile;
        myfile.open("../img.bin", ios::out|ios::binary|ios::ate);
        char* pData = (char *) malloc(512000);
        myfile.read(pData, 512000);
        myfile.close();
        query_result qr = query_list(handler, pData, 640, 400, 2);
        cout << "Success, query_result =  " << qr.get_id << "confidence = " << qr.confidence << endl;
}
