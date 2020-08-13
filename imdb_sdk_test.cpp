#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
#include "tic_toc.h"
#include "imdb_sdk.h"
#include <opencv2/opencv.hpp>
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


int main(int argc, char *argv[])
{
    int addStep = 27;//135;//
    int queryStep = 15;//down sample FPS = 30/(queryStep+1) fps
    int query_list_num = 10;//query period = query_list_num/FPS
    const int IMG_WIDTH = 640;
    const int IMG_HEIGHT = 400;
    string voc_path= "../config/loopC_vocdata.bin";
    string fileListsPath = "../config/datalist2.txt";
    string testListsPath = "../config/datalist3.txt";
    std::string pattern_file = "../config/loopC_pattern.yml";
#if 0
    pattern_file = "/home/what/disk/works/image_retrieval/loopC_pattern.yml";
    voc_path = "/home/what/disk/works/image_retrieval/loopC_vocdata.bin";
    fileListsPath = "../config/dlist2.txt";
    testListsPath = "../config/dlist3.txt";
#endif
//    ImageDatabase imdb(voc_path, pattern_file);
    void *handler1 = initDataBase(voc_path, pattern_file);
    vector<string> fileVec;
    LoadPathList(fileListsPath, fileVec);
    TicToc t_loadImage;
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
            std::cout << "The size of image list is " << imagesList.size() << " downsample to :" <<
                      imagesList.size() / addStep << endl;
        }
        bool skip = false;
        try {
            if(argc >= 2 ) {
                for(int j = 1; j < argc; j++) {
                    if(i == stoi(argv[j])){
                        cout << "skip set:" << fileVec[i] << endl;
                        skip = true;
                        break;
                    }
                }
                if(skip) continue;
            }
        }
        catch (...)
        {}
        for (int ni = 0; ni < imagesList.size(); ni += addStep) {
              addImage(handler1,imagesList[ni],i);
        }
        counterDb += imagesList.size() / addStep;
    }
    loadImageTimeCost = t_loadImage.toc();
    string fileListsPathQuery = testListsPath;
    LoadPathList(fileListsPathQuery, fileVec);
    vector<int> counter1,counter2,counter3;
    TicToc t_queryImage;
//    char image_data[IMG_WIDTH*IMG_HEIGHT*query_list_num];
    // char* image_data =(char *) malloc(IMG_WIDTH*IMG_HEIGHT*query_list_num);
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
            std::cout << "The size of image list is " << imagesList.size() << endl;
        }

        int add_cycle = 0;
        char image_data[IMG_WIDTH*IMG_HEIGHT*query_list_num];
        char * image_data_ptr = image_data;
        int img_size = IMG_WIDTH*IMG_HEIGHT;
        for (int ni = 0; ni < imagesList.size(); ni += queryStep) {
            cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
            cv::Mat im_rsz = image.reshape(0,1);
            if(add_cycle<query_list_num)
            {
                memcpy(image_data_ptr, im_rsz.data, img_size);
                image_data_ptr += img_size;
                add_cycle++;
            }
            else
            {
                query_result qr = query_list(handler1, image_data, IMG_WIDTH, IMG_HEIGHT, query_list_num);
                int id = qr.set_id;
                double confidence = qr.confidence;
                counter1[i]++;
                if(id == i) counter2[i]++;
                if(id == -1) counter3[i]++;
                std::cout << "The setid and result is: " << i << " - " << id << "(" << confidence << ")\n" << endl;
                add_cycle=0;
                image_data_ptr = image_data;
//                return 0;
            }
        }
        counterQuery += imagesList.size() / queryStep;
    }
    // free(image_data);
    queryImageTimeCost = t_loadImage.toc();
    try {
        if (argc >= 2) {
            for (int j = 1; j < argc; j++) {
                int index = stoi(argv[j]);
                std::cout << "---***set id: " << index << endl;
                std::cout << "---***The match number is: " << counter2[index] << endl;
                std::cout << "---***The failed number is: " << counter3[index] << endl;
                std::cout << "---***The total number is: " << counter1[index] << endl;
                std::cout << "---***Success rate is: " << 1.0f * counter2[index] / counter1[index] << endl << endl;
            }
        }
    }
    catch (...)
    {}
    for(auto i = 0; i < fileVec.size(); i++) {
        std::cout << "***set id: " << i << endl;
        std::cout << "***The match number is: " << counter2[i] << endl;
        std::cout << "***The failed number is: " << counter3[i] << endl;
        std::cout << "***The total number is: " << counter1[i] << endl;
        std::cout << "***Success rate is: " << 1.0f * counter2[i] / counter1[i] << endl << endl;
    }
    std::cout << "loadImageTimeCost(s) is: " << loadImageTimeCost / 1000 << endl;
    std::cout << "queryImageTimeCost(s) is: " << queryImageTimeCost / 1000 << endl;
    std::cout << "loadImageTotalNumber is: " << counterDb << endl;
    std::cout << "queryImageTotalNumber is: " << counterQuery << endl;
    std::cout << "avg loadImageTimeCost(ms) is: " << loadImageTimeCost / counterDb<< endl;
    std::cout << "avg queryImageTimeCost(ms) is: " << queryImageTimeCost / counterQuery << endl;
}