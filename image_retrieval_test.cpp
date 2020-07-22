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

int create_db = 0;
const int scene_num=10;
const int vote_window=30;
std::vector<pair<int,int>> vote;
queue<int> id_que;
int last_id[3],raising_cnt;

int main()
{
    int addStep = 10;
    int queryStep = 10;

    for (size_t i = 0; i < 10; i++) vote.push_back(make_pair(i,0));

    // string voc_path= "/home/wankai/project/image_retrieval/config/loopC_vocdata.bin";
    // string fileListsPath = "../config/datalist1.txt";
    // std::string pattern_file = "/home/wankai/project/image_retrieval/config/loopC_pattern.yml";

    string pattern_file = "/home/what/disk/works/image_retrieval/loopC_pattern.yml";
    string voc_path = "/home/what/disk/works/image_retrieval/loopC_vocdata.bin";
    string fileListsPath = "../config/datalist2.txt";

    ImageDatabase imdb(voc_path, pattern_file);
    vector<string> fileVec;
    LoadPathList(fileListsPath, fileVec);
    TicToc t_loadImage;
    double loadImageTimeCost, queryImageTimeCost;
    int counterDb = 0, counterQuery = 0;
    vector<BRIEF::bitset> brief_descriptors;
    vector<pair<int,vector<string>>> images_DBList;
    for(auto i = 0; i < fileVec.size(); i++) {
        string base_path = fileVec[i];
        string image_path = base_path + "/cam0";
        string timeStamps = base_path + "/loop.txt";
        vector<string> imagesList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, imagesList);
            cout << "The size of image list is " << imagesList.size() << ",  image_path:" << image_path << endl;
        }

        images_DBList.push_back(make_pair(i,imagesList));

        for (int ni = 0; ni < imagesList.size(); ni += addStep) {
            cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
            #ifndef DEBUG
            imdb.extractFeatureVector(image, brief_descriptors);
            imdb.addImage(image, i);
            #else
            imdb.addImage(image, i, ni, create_db);
            #endif
        }
        counterDb += imagesList.size() / addStep;
    }
    loadImageTimeCost = t_loadImage.toc();

    #ifdef DEBUG
    if(create_db==0) {
        printf("loading db...\n");
        imdb.db.load("image.db");
    }
    else imdb.db.save("image.db");
    #endif

    string fileListsPathQuery = "../config/datalist3.txt";
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
        vector<string> images_QRList;
        if (!image_path.empty()) {
            LoadImages(image_path, timeStamps, images_QRList);
            cout << "The size of image list is " << images_QRList.size() << endl;
        }
        for (int ni = 0; ni < images_QRList.size(); ni += queryStep) {
            cv::Mat image = cv::imread(images_QRList[ni], CV_LOAD_IMAGE_UNCHANGED);
            #ifndef DEBUG
            int id = imdb.query(image);
            #else
            pair<int,int> id = imdb.query(image);
            #endif

            //-----vote
            vote[id.first].second++;
            id_que.push(id.first);
            int most_trust=-1;
            int vote_id=-1, trust_id=-1;
            std::vector<pair<int,int>> vote_clone=vote;
        
            sort(vote_clone.begin(), vote_clone.end(),
                [](const pair<int, int> &a, const pair<int, int> &b) {
                return a.second > b.second;
            });

            if(id_que.size()>(vote_window-1))
            {
                vote[id_que.front()].second--;
                id_que.pop();
                for (size_t i = 0; i < vote_clone.size(); i++)
                {
                    printf("sort vote:%d,%d\n",vote_clone[i].first,vote_clone[i].second);
                }
                double confidence1 = (double)vote_clone[0].second/(double)vote_window;
                double confidence2 = (double)vote_clone[1].second/(double)vote_window;

                if( confidence1 > 0.50)
                {
                    trust_id=vote_clone[0].first;
                    printf("the most recommend scene id: %d(%f)\n", trust_id,confidence1);
                }
                else
                {
                    printf("the recommend two scene ids: first-%d(%f) second-%d(%f)\n", vote_clone[0].first,confidence1,vote_clone[1].first,confidence2);
                }
            }
            else printf("initializing\n");

            if(last_id[0]!=trust_id&&last_id[0]==last_id[1]&&last_id[1] == last_id[2]) raising_cnt++;
            else raising_cnt=0;
            printf("vote_first:%d, last_id:%d,%d,%d, raising_cnt:%d\n",vote_clone[0].first, last_id[0],last_id[1],last_id[2], raising_cnt);
            if(raising_cnt>5) printf("~~~~~~~~~~~~~~~~~you may be entering scene id: %d~~~~~~~~~~~~~~~~~\n", last_id[0]);
            last_id[2] = last_id[1];
            last_id[1] = last_id[0];
            last_id[0] = id.first;
            // -----vote end

            #ifdef DEBUG
            if(id.first != -1)
            {
                cv::Mat imageDB = cv::imread(images_DBList[id.first].second[id.second], CV_LOAD_IMAGE_UNCHANGED);
                cv::Mat img, show_img;
                // imgSIFT(image, imageDB);
                // imgSURF(image, imageDB);
                cv::hconcat(image, imageDB, show_img);
                // img.convertTo(show_img, CV_8UC3);
                // cvCvtColor(&img, &show_img, cv::COLOR_GRAY2RGB);
                char buf1[10],buf2[10];
                sprintf(buf1, "test: %d-%d", i, ni);
                cv::putText(show_img, buf1, cv::Point(10,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
                sprintf(buf2, "query: %d-%d", id.first,id.second);
                cv::putText(show_img, buf2, cv::Point(650,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
                cv::imshow("matched", show_img);
                cv::waitKey(0);
            }
            #endif

            counter1[i]++;
            #ifndef DEBUG
            if(id == i) counter2[i]++;
            if(id == -1) counter3[i]++;
            cout << "The setid and result is: " << i << " - " << id << endl;
            #else
            if(id.first == i) counter2[i]++;
            else if(id.first == -1) counter3[i]++;
            cout << "The setid and result is: " << i << "(" << ni << ")" << " - " << id.first << "(" << id.second << ")"<< endl;
            #endif
        }
        counterQuery += images_QRList.size() / queryStep;
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