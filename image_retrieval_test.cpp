#include <string>
#include <iostream>
//using namespace std;
#include "brief.h"


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
void blurImage4Brief(cv::Mat const &src, cv::Mat &dst){
    if (!src.empty())
    {
        const float sigma = 2.f;
        const cv::Size ksize(9, 9);

        cv::GaussianBlur(src, dst, ksize, sigma, sigma);
    }
}

void computeBRIEFPoint(cv::Mat &image, cv::Mat &image_blur,vector<cv::KeyPoint> &keypoints,
        vector<BRIEF::bitset> &brief_descriptors, std::string pattern_file)
{

    static BriefExtractor m_extractor;
    m_extractor.init(pattern_file.c_str());
    int fast_th = 20; // corner detector response threshold
    //cv::Mat tmp_img;
    //cv::GaussianBlur(image, tmp_img, cv::Size(5, 5), 1.2, 0, cv::BORDER_DEFAULT);

    vector<cv::Point2f> tmp_pts;
    int MAX_KEYFRAME_KEYPOINTS = 1000;
    cv::goodFeaturesToTrack(image, tmp_pts, MAX_KEYFRAME_KEYPOINTS, 0.01, 3);
    for(int i = 0; i < (int)tmp_pts.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = tmp_pts[i];
        keypoints.push_back(key);
    }

    LOGD("FAST keypoints size = %zu, fast_th = %d", keypoints.size(), fast_th);
    m_extractor(image_blur, keypoints, brief_descriptors);
}
int main()
{
    std::string voc_path= "/home/wankai/project/image_retrieval/config/loopC_vocdata.bin";
    BriefVocabulary* voc = new BriefVocabulary(voc_path);
    BriefDatabase db;
    if(voc == NULL)
    {
        printf("****** vocabulary load failed, loop can not detect");
    }
    db.setVocabulary(*voc, true, 0);
    string base_path = "/home/wankai/data/slam/slamdata-Chicken/slamdata-circle-01/";
    string image_path = base_path + "cam0";
    string timeStamps = base_path + "loop.txt";
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
        cv::Mat image_blur;
        blurImage4Brief(image, image_blur);
//        cv::imshow("image", image);
//        cv::imshow("image_blur", image_blur);
//        cv::waitKey(10);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        std::string pattern_file = "/home/wankai/project/image_retrieval/config/loopC_pattern.yml";
        computeBRIEFPoint(image,image_blur,keypoints,brief_descriptors, pattern_file);
        db.add(brief_descriptors);
    }
    for (int ni = 0; ni < imagesList.size(); ni++) {
        cv::Mat image = cv::imread(imagesList[ni], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat image_blur;
        blurImage4Brief(image, image_blur);
//        cv::imshow("image", image);
//        cv::imshow("image_blur", image_blur);
//        cv::waitKey(10);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        std::string pattern_file = "/home/wankai/project/image_retrieval/config/loopC_pattern.yml";
        computeBRIEFPoint(image,image_blur,keypoints,brief_descriptors, pattern_file);
        DBoW2::QueryResults ret;
        db.query(brief_descriptors, ret, 4, imagesList.size());
        if (ret.size() >= 1 && ret[0].Score > 0.05) printf("Query Result: input id: %d, query result: %d \n", ni, ret[0].Id);
    }
//    BriefVocabulary* voc = new BriefVocabulary();
    delete(voc);

}