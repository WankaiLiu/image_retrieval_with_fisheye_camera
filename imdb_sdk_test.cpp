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
    int queryStep = 15;//down sample FPS = 30/(queryStep+1) fps

    cout << "\n============1.Set up database=================" << endl;
    void *handler1 = initDataBase(path_to_config_voc, path_to_config_ptn);
    vector<pair<string,int>> file_id_list;
    LoadPathIdList(path_to_config_dbase, file_id_list);
    string path_to_config_dbaseQuery = path_to_config_tlist;
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
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            std::cout << "The size of image database list(id:" << set_id << ") is " << imagesList.size() \
 << " downsample to :" << imagesList.size() / addStep << endl;
#ifdef DEBUGD
            saveImagePath(handler1, i,imagesList);
#endif
        }
        for (int ni = 0; ni < imagesList.size(); ni += addStep) {
#ifdef DEBUGD
            addImage(handler1,imagesList[ni], set_id, "/home/wankai/data/slam/slamdata-Chicken/sunflower/cam0_NG2-sdm845.yaml");
#else
            addImage(handler1, imagesList[ni], set_id,
                     "/home/wankai/data/slam/slamdata-Chicken/sunflower/cam0_NG2-sdm845.yaml");
#endif
        }
        counterDb += imagesList.size() / addStep;
    }
    loadImageTimeCost = t_loadImage.toc();

    cout << "\n============2.Start to query======================" << endl;



    vector<pair<string,int>> file_id_list_query;
    LoadPathIdList(path_to_config_tlist, file_id_list_query);
    unordered_map<int, int> counter1,counter2,counter3;
    TicToc t_queryImage;
    int count_cycle = 0;
    std::vector<std::string> query_vec;
    std::vector<query_result> query_result_vec;
    for(auto i = 0; i < file_id_list_query.size(); i++) {
        string base_path = file_id_list_query[i].first;
        int set_id = file_id_list_query[i].second;
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            std::cout << "The size of image query list(id:"<< set_id << ") is " << imagesList.size() \
                        << " downsample to :" << imagesList.size() / queryStep << endl;
#ifdef DEBUGD
            saveImagePath(handler1, i,imagesList);
#endif
        }
        query_result_vec = query_list_vec(handler1, query_vec,"/home/wankai/data/slam/slamdata-Chicken/sunflower/cam0_NG2-sdm845.yaml");
        for (int ni = 0; ni < imagesList.size(); ni += queryStep) {
            if(count_cycle != 10) {
                query_vec.push_back(imagesList[ni]);
                count_cycle++;
            }
            else{
                query_result_vec = query_list_vec(handler1, query_vec,"/home/wankai/data/slam/slamdata-Chicken/sunflower/cam0_NG2-sdm845.yaml");
                for (int ret_i = 0; ret_i < query_result_vec.size(); ret_i++) {
                    cout << "querying result: " << query_result_vec[ret_i].get_id << "--" <<
                         query_result_vec[ret_i].confidence << endl;
                }
                query_vec.clear();
                count_cycle = 0;
            }
        }
        query_vec.clear();
        count_cycle = 0;

        counterQuery = file_id_list_query.size();

    }

    cout << "\n============3.Test erase set function======================" << endl;
    for(auto i = 0; i < file_id_list_query.size(); i++) {

        //CLEAR DATASET

        string base_path = file_id_list_query[i].first;
        int qr_id = file_id_list_query[i].second;

        TicToc tquery;
        bool status = erase_set(handler1, qr_id);
        cout << "erase current set: " << qr_id << " , with status" << status << endl;
        int set_id = file_id_list_query[i].second;
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            std::cout << "The size of image query list(id:"<< set_id << ") is " << imagesList.size() \
                        << " downsample to :" << imagesList.size() / queryStep << endl;
#ifdef DEBUGD
            saveImagePath(handler1, i,imagesList);
#endif
        }
        query_result_vec = query_list_vec(handler1, query_vec,"/home/wankai/data/slam/slamdata-Chicken/sunflower/cam0_NG2-sdm845.yaml");
        for (int ni = 0; ni < imagesList.size(); ni += queryStep) {
            if(count_cycle != 10) {
                query_vec.push_back(imagesList[ni]);
                count_cycle++;
            }
            else{
                query_result_vec = query_list_vec(handler1, query_vec,"/home/wankai/data/slam/slamdata-Chicken/sunflower/cam0_NG2-sdm845.yaml");
                for (int ret_i = 0; ret_i < query_result_vec.size(); ret_i++) {
                    cout << "querying result: " << query_result_vec[ret_i].get_id << "--" <<
                         query_result_vec[ret_i].confidence << endl;
                }
                query_vec.clear();
                count_cycle = 0;
            }
        }
        query_vec.clear();
        count_cycle = 0;
        for (int ni = 0; ni < imagesList.size(); ni += addStep) {
#ifdef DEBUGD
            addImage(handler1,imagesList[ni], qr_id, ni);
#else
            addImage(handler1,imagesList[ni], qr_id,"/home/wankai/data/slam/slamdata-Chicken/sunflower/cam0_NG2-sdm845.yaml");
#endif
        }
    }

    for (auto &id : counter1){
        std::cout << "***set id: " << id.first << endl;
        std::cout << "***The match number is: " << counter2[id.first] << endl;
        std::cout << "***The failed number is: " << counter3[id.first] << endl;
        std::cout << "***The total number is: " << counter1[id.first] << endl;
        std::cout << "***Success rate is: " << 1.0f * counter2[id.first] / id.second << endl << endl;
    }
    std::cout << "loadImageTimeCost(s) is: " << loadImageTimeCost / 1000 << endl;
    std::cout << "queryImageTimeCost(s) is: " << queryImageTimeCost / 1000 << endl;
    std::cout << "loadImageTotalNumber is: " << counterDb << endl;
    std::cout << "queryImageTotalNumber is: " << counterQuery << endl;
    std::cout << "avg loadImageTimeCost(ms) is: " << loadImageTimeCost / counterDb<< endl;
    std::cout << "avg queryImageTimeCost(ms) is: " << queryImageTimeCost / counterQuery << endl;
}