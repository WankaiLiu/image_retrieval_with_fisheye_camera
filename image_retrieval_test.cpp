#include <string>
#include <iostream>
//using namespace std;
#include "tic_toc.h"
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
    string voc_path= "/home/wankai/project/image_retrieval/config/loopC_vocdata.bin";
    string fileListsPath = "../config/datalist2.txt";
    std::string pattern_file = "/home/wankai/project/image_retrieval/config/loopC_pattern.yml";
    ImageDatabase imdb(voc_path, pattern_file);
    vector<string> fileVec;
    LoadPathList(fileListsPath, fileVec);
    TicToc t_loadImage;
    double loadImageTimeCost, queryImageTimeCost;
    int counterDb = 0, counterQuery = 0;
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
            cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
            imdb.addImage(image, i);
        }
        counterDb += imagesList.size() / addStep;
    }
    loadImageTimeCost = t_loadImage.toc();
    string fileListsPathQuery = "../config/datalist1.txt";
    LoadPathList(fileListsPathQuery, fileVec);
    vector<int> counter1,counter2,counter3;
    TicToc t_queryImage;
    for(auto i = 0; i < fileVec.size(); i++) {
        counter1.push_back(0);
        counter2.push_back(0);
        counter3.push_back(0);
        string base_path = fileVec[i];
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            cout << "The size of image list is " << imagesList.size() << endl;
        }
        for (int ni = 0; ni < imagesList.size(); ni += queryStep) {
            cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
            int id = imdb.query(image);
            counter1[i]++;
            if(id == i) counter2[i]++;
            if(id == -1) counter3[i]++;
            cout << "The setid and result is: " << i << " - " << id << endl;
        }
        counterQuery += imagesList.size() / queryStep;
    }
    queryImageTimeCost = t_loadImage.toc();
    for(auto i = 0; i < fileVec.size(); i++) {
        cout << "set id: " << i << endl;
        cout << "The match number is: " << counter2[i] << endl;
        cout << "The failed number is: " << counter3[i] << endl;
        cout << "The total number is: " << counter1[i] << endl;
        cout << "Success rate is: " << 1.0f * counter2[i] / counter1[i] << endl;
    }
    cout << "loadImageTimeCost(s) is: " << loadImageTimeCost / 1000 << endl;
    cout << "queryImageTimeCost(s) is: " << queryImageTimeCost / 1000 << endl;
    cout << "loadImageTotalNumber is: " << counterDb << endl;
    cout << "queryImageTotalNumber is: " << counterQuery << endl;
    cout << "avg loadImageTimeCost(ms) is: " << loadImageTimeCost / counterDb<< endl;
    cout << "avg queryImageTimeCost(ms) is: " << queryImageTimeCost / counterQuery << endl;
}