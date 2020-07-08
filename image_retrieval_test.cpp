#include <string>
#include <iostream>
//using namespace std;
//#include "brief.h"
#include "ImageDatabase.h"


void LoadImages(const string &strImagePath, const string &strTimesStampsPath,
                vector<string> &strImagesFileNames)
{
    std::cout << "LoadImages(): " << strTimesStampsPath << std::endl;
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



int main()
{

    std::string voc_path= "/home/wankai/project/image_retrieval/config/loopC_vocdata.bin";


    string base_path = "/home/wankai/data/slam/slamdata-Chicken/slamdata-circle-01/";
    string image_path = base_path + "cam0";
    string timeStamps = base_path + "loop.txt";
    std::string pattern_file = "/home/wankai/project/image_retrieval/config/loopC_pattern.yml";
    ImageDatabase imdb(voc_path, pattern_file);
    vector<string> imagesList;
//    ./estimator_test -c=/home/wankai/data/slam/slamdata-Chicken/sunflower/config_g2.yaml -l=$DATA_DIR/$DATA/cam0
//            -r=$DATA_DIR/$DATA/cam1 -ts=$DATA_DIR/$DATA/loop.txt -imu=$DATA_DIR/$DATA/imu0.csv
    if(!image_path.empty()) {

        LoadImages(image_path, timeStamps, imagesList);
        cout << "The size of image list is " << imagesList.size() << endl;
    }
//    for (int ni = 0; ni < imagesList.size(); ni++) {
    for (int ni = 0; ni < 10; ni++) {
        cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
        imdb.addImage(image, ni%3);
    }
    for (int ni = 0; ni < imagesList.size(); ni++) {
        cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
        int id = imdb.query(image);
        cout << "The result is " << id << endl;
    }
//    BriefVocabulary* voc = new BriefVocabulary();

}