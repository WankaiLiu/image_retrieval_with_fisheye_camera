#include <string>
#include <iostream>
//using namespace std;
//#include "brief.h"
#include "ImageDatabase.h"


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
    string voc_path= "/home/wankai/project/image_retrieval/config/loopC_vocdata.bin";
    string fileListsPath = "../config/datalist1.txt";
    std::string pattern_file = "/home/wankai/project/image_retrieval/config/loopC_pattern.yml";
    ImageDatabase imdb(voc_path, pattern_file);
    vector<string> fileVec;
    LoadPathList(fileListsPath, fileVec);
    for(auto i = 0; i < fileVec.size(); i++) {
        string base_path = fileVec[i];
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            cout << "The size of image list is " << imagesList.size() << endl;
        }
        for (int ni = 0; ni < 10; ni++) {
            cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
            imdb.addImage(image, i);
        }
    }
    for(auto i = 0; i < fileVec.size(); i++) {
        string base_path = fileVec[i];
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            cout << "The size of image list is " << imagesList.size() << endl;
        }
        for (int ni = 0; ni < imagesList.size(); ni++) {
            cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
            int id = imdb.query(image);
            cout << "The result is " << id << endl;
        }
    }

}