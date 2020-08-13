#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
#include "tic_toc.h"
#include "imdb_sdk.h"
#include <opencv2/opencv.hpp>
#define DEBUG_INFO 0

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

void LoadPathIdList( const string &fileListsPath, vector<pair<string,int>> &fileVec)
{
    fileVec.clear();
    if(DEBUG_INFO) std::cout << "LoadImages(): " << fileListsPath << std::endl;
    ifstream fTimes;
    fTimes.open(fileListsPath.c_str());
    while (!fTimes.eof())
    {
        string s;
        char file_path[1024]={0};
        int set_id;
        getline(fTimes, s);
        if (!s.empty())
        {
            // cout << s <<endl;
            sscanf(s.c_str(), "%s %d", file_path, &set_id);
            // cout << file_path << "   " << set_id<<endl;
            stringstream ss;
            ss << file_path;
            if(DEBUG_INFO) std::cout << "load set: " << ss.str() << std::endl;

            fileVec.push_back(make_pair(ss.str(),set_id));
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

#define IMG_WIDTH 640
#define IMG_HEIGHT 400
#define Q_LIST_NUM 10

//   ./imdb_sdk_test \
//   -voc=/home/what/disk/works/image_retrieval/config/loopC_vocdata.bin \
//   -ptn=/home/what/disk/works/image_retrieval/config/loopC_pattern.yml \
//   -dbase=/home/what/disk/works/image_retrieval/config/dlist2.txt \
//   -tlist=/home/what/disk/works/image_retrieval/build2/query_list.txt \
//   -mode=1

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, "{voc||}{ptn||}{dbase||}{tlist||}");

    std::string path_to_config_voc;
    if (parser.has("voc"))  path_to_config_voc = parser.get<string>("voc");
    if(path_to_config_voc.empty())
    {
        cout << "ERROR: please give the path of loopC_vocdata.bin, add it like: -voc=YOUR_PATH" << endl;
    }

    std::string path_to_config_ptn;
    if (parser.has("ptn"))  path_to_config_ptn = parser.get<string>("ptn");
    if(path_to_config_ptn.empty())
    {
        cout << "ERROR: please give the path of loopC_pattern.yml, add it like: -ptn=YOUR_PATH" << endl;
    }

    std::string path_to_config_dbase;
    if (parser.has("dbase"))  path_to_config_dbase = parser.get<string>("dbase");
    if(path_to_config_dbase.empty())
    {
        cout << "ERROR: please give the path of your image database, add it like: -dbase=YOUR_PATH" << endl;
    }

    std::string path_to_config_tlist;
    if (parser.has("tlist"))  path_to_config_tlist = parser.get<string>("tlist");
    if(path_to_config_tlist.empty())
    {
        cout << "ERROR: please give the path of your test images, add it like: -tlist=YOUR_PATH" << endl;
    }

    if(path_to_config_voc.empty() || path_to_config_ptn.empty() || path_to_config_dbase.empty() || path_to_config_tlist.empty())
    {
        cout << "Parameter is not enough!" << endl;
        return -1;
    }
    else
    {
        cout << "voc      : " << path_to_config_voc << endl;
        cout << "pattern  : " << path_to_config_ptn << endl;
        cout << "database : " << path_to_config_dbase << endl;
        cout << "testlist : " << path_to_config_tlist << endl;
    }

    cout << "\nstart evaluating" << endl;

    int addStep = 27;//135;//
    int queryStep = 1;//down sample FPS = 30/(queryStep+1) fps

    cout << "\n1.set up database" << endl;
    void *handler1 = initDataBase(path_to_config_voc, path_to_config_ptn);
    vector<pair<string,int>> file_id_list;
    LoadPathIdList(path_to_config_dbase, file_id_list);
    TicToc t_loadImage;
    double loadImageTimeCost, queryImageTimeCost;
    int counterDb = 0, counterQuery = 0;
    int jump_cnt = 5;
    for(auto i = 0; i < file_id_list.size(); i++) {
        string base_path = file_id_list[i].first;
        int set_id = file_id_list[i].second;
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        {//jump
            bool skip = false;
            try {
                if(argc >= 6 && jump_cnt < argc) {
                    for(int j = 5; j < argc; j++) {
                        if(set_id == stoi(argv[j])){ //跳过set_id
                            cout << ">>>  skip set(id:"<< set_id << ")" << base_path << endl;
                            skip = true;
                            jump_cnt++;
                            break;
                        }
                    }
                    if(skip) continue;
                }
            }
            catch (...)
            {}
        }
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            std::cout << "The size of image database list(id:"<< set_id << ") is " << imagesList.size() \
                        << " downsample to :" << imagesList.size() / addStep << endl;
            #ifdef DEBUGD
            saveImagePath(handler1, i,imagesList);
            #endif
        }
        for (int ni = 0; ni < imagesList.size(); ni += addStep) {
            #ifdef DEBUGD
            addImage(handler1,imagesList[ni], set_id, ni);
            #else
            addImage(handler1,imagesList[ni], set_id);
            #endif
        }
        counterDb += imagesList.size() / addStep;
    }
    loadImageTimeCost = t_loadImage.toc();

    cout << "\n2.query" << endl;
    file_id_list.clear();
    string path_to_config_dbaseQuery = path_to_config_tlist;
    LoadPathIdList(path_to_config_tlist, file_id_list);
    vector<int> counter1,counter2,counter3;
    TicToc t_queryImage;
    char image_list_bin[IMG_WIDTH*IMG_HEIGHT*Q_LIST_NUM];
    for(auto i = 0; i < file_id_list.size(); i++) {
        counter1.push_back(0);
        counter2.push_back(0);
        counter3.push_back(0);
        string base_path = file_id_list[i].first;
        int qr_id = file_id_list[i].second;
        // cout << base_path << qr_id << endl;
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if(!image_path.empty()) {
            LoadPathList(base_path, imagesList);//query mode
            std::cout << "The size of image list is " << imagesList.size() << endl;
        }

        char image_data[IMG_WIDTH*IMG_HEIGHT*Q_LIST_NUM];
        char * image_data_ptr = image_data;
        int img_size = IMG_WIDTH*IMG_HEIGHT;
        int add_cycle = 0,tmp=0;
        for (int ni = 0; ni < imagesList.size(); ni ++) {
            cout << "querying:  "<<imagesList[ni].c_str()<<endl;
            FILE* fp = fopen(imagesList[ni].c_str(),"rb");
            if(fp)
            {
                fread(image_list_bin,sizeof(char),IMG_WIDTH*IMG_HEIGHT*Q_LIST_NUM,fp);
                fclose(fp);
                fp = NULL;
            }
            else return -1;

            TicToc tquery;
            #ifdef DEBUGD
            query_result qr = query_list(handler1, i, image_list_bin, IMG_WIDTH, IMG_HEIGHT, Q_LIST_NUM);
            #else
            query_result qr = query_list(handler1, image_list_bin, IMG_WIDTH, IMG_HEIGHT, Q_LIST_NUM);
            #endif
            int get_id = qr.get_id;
            double confidence = qr.confidence;
            counter1[i]++;
            if(get_id == qr_id) counter2[i]++;
            if(get_id == -1) counter3[i]++;
            std::cout << "The setid and result is: " << qr_id << " - " << get_id << "(" << confidence << "),  cost time:" << tquery.toc() <<"ms\n" << endl;
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
    for(auto i = 0; i < file_id_list.size(); i++) {
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