#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <stdio.h>
using namespace std;
#include "tic_toc.h"
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
            cout << file_path << "   id:" << set_id<<endl;
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
#define MAXPATH 1024
//   ./imdb_sdk_test \
//   -voc=/home/what/disk/works/image_retrieval/config/loopC_vocdata.bin \
//   -ptn=/home/what/disk/works/image_retrieval/config/loopC_pattern.yml \
//   -dbase=/home/what/disk/works/image_retrieval/config/dlist2.txt \
//   -tlist=/home/what/disk/works/image_retrieval/build2/query_list.txt \
//   -mode=1

int main(int argc, char *argv[])
{
    string command = "rm list*.txt";
    FILE * ptr = NULL;
    if (NULL == (ptr = popen(command.c_str(),"r")))
    {
        cout << "failed to clear old list*.txt\n";
        return -1;
    }
    command = "rm query_list.txt";
    if (NULL == (ptr = popen(command.c_str(),"r")))
    {
        cout << "failed to clear old query_list.txt\n";
        return -1;
    }
    command = "rm -rf bindata && mkdir bindata";
    if (NULL == (ptr = popen(command.c_str(),"r")))
    {
        cout << "failed to clear bindata\n";
        return -1;
    }
    pclose(ptr);
    cv::CommandLineParser parser(argc, argv,
        "{tlist||}");
    std::string path_to_config_tlist;
    if (parser.has("tlist"))  path_to_config_tlist = parser.get<string>("tlist");
    if(path_to_config_tlist.empty())
    {
        cout << "ERROR: please give the path of your test images, add it like: -tlist=YOUR_PATH" << endl;
    }

    string mode = "0";
    if(path_to_config_tlist.empty())
    {
        cout << "Parameter is not enough!" << endl;
        {
    cout << "open file error" << endl;
    return -1;
}
    }
    else
    {
        cout << "testlist : " << path_to_config_tlist << endl;
    }

    // cout << "start evaluating" << endl;

    int addStep = 27;//135;//
    int queryStep = 1;//down sample FPS = 30/(queryStep+1) fps

    cout << "start build bin-database" << endl;
    // void *handler1 = initDataBase(path_to_config_voc, path_to_config_ptn);
    vector<pair<string,int>> file_id_list;
    // LoadPathIdList(path_to_config_dbase, file_id_list);
    TicToc t_loadImage;
    double loadImageTimeCost, queryImageTimeCost;
    int counterDb = 0, counterQuery = 0;
    char cur_path[MAXPATH];
    getcwd(cur_path,MAXPATH);

    file_id_list.clear();
    string path_to_config_dbaseQuery = path_to_config_tlist;
    LoadPathIdList(path_to_config_tlist, file_id_list);
    char image_list_bin[IMG_WIDTH*IMG_HEIGHT*Q_LIST_NUM];
    for(auto i = 0; i < file_id_list.size(); i++) {
        string base_path = file_id_list[i].first;
        int qr_id = file_id_list[i].second;
        // cout << base_path << qr_id << endl;
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if(!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);//genery bin data mode
            std::cout << "The size of image list is " << imagesList.size()  ;
        }

        char image_data[IMG_WIDTH*IMG_HEIGHT*Q_LIST_NUM];
        char * image_data_ptr = image_data;
        int img_size = IMG_WIDTH*IMG_HEIGHT;
        int add_cycle = 0,tmp=0;
        char list_path[1024];
        cout << ",       converting image_set ID:"<< qr_id << " to binary files" <<endl;
        for (int ni = 0; ni < imagesList.size(); ni ++) {
                cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
                cv::Mat im_rsz = image.reshape(0,1);
                if(add_cycle<Q_LIST_NUM)
                {
                    memcpy(image_data_ptr, im_rsz.data, img_size);
                    image_data_ptr += img_size;
                    add_cycle++;
                }
                else
                {
                    char buf[1024];
                    sprintf(buf,"bindata/%d-%d.bin",qr_id,tmp);
                    string save_bin_path = (string)cur_path+"/"+(string)buf;
                    FILE* fp = fopen(save_bin_path.c_str(),"wb+");//binary
                    // cout << "GEN_IMG_BIN_DATA  " << buf<<endl;
                    if(fp)
                    {
                        fwrite(image_data,sizeof(char),IMG_WIDTH*IMG_HEIGHT*Q_LIST_NUM,fp);
                        fclose(fp);
                        fp = NULL;
                    }
                    else {
                        cout << "create " << save_bin_path <<  " error" << endl;
                        return -1;
                    }
                    sprintf(list_path,"list%d.txt",qr_id);
                    // sprintf(buf,"bindata/%d-%d.bin\n",qr_id,tmp);
                    save_bin_path += "\n";
                    FILE* flist = fopen(list_path,"a+");//binary
                    if(flist)
                    {
                        fwrite(save_bin_path.c_str(),sizeof(char),save_bin_path.size(),flist);
                        fclose(flist);
                        flist = NULL;
                    }
                    else {
                        cout << "create " << list_path <<  " error" << endl;
                        return -1;
                    }
                    add_cycle=0;
                    image_data_ptr = image_data;
                    tmp++;
                    ni += 13;
                }
        }
        counterQuery += imagesList.size() / queryStep;
        sprintf(list_path,"list%d.txt %d\n",qr_id,qr_id);
        FILE* fqlist = fopen("query_list.txt","a+");//binary
        if(fqlist)
        {
            fwrite(list_path,sizeof(char),string(list_path).size(),fqlist);
            fclose(fqlist);
            fqlist = NULL;
        }
        else {
            cout << "create query_list.txt error" << endl;
            return -1;
        }
    }
    // free(image_data);
    // queryImageTimeCost = t_loadImage.toc();

    // cout << "\n\n3.display result" << endl;
    // for(auto i = 0; i < file_id_list.size(); i++) {
    //     std::cout << "set id: " << i << endl;
    //     std::cout << "The match number is: " << counter2[i] << endl;
    //     std::cout << "The failed number is: " << counter3[i] << endl;
    //     std::cout << "The total number is: " << counter1[i] << endl;
    //     std::cout << "Success rate is: " << 1.0f * counter2[i] / counter1[i] << endl << endl;
    // }
    // std::cout << "loadImageTimeCost(s) is: " << loadImageTimeCost / 1000 << endl;
    // std::cout << "queryImageTimeCost(s) is: " << queryImageTimeCost / 1000 << endl;
    // std::cout << "loadImageTotalNumber is: " << counterDb << endl;
    // std::cout << "queryImageTotalNumber is: " << counterQuery << endl;
    // std::cout << "avg loadImageTimeCost(ms) is: " << loadImageTimeCost / counterDb<< endl;
    // std::cout << "avg queryImageTimeCost(ms) is: " << queryImageTimeCost / counterQuery << endl;
}