//
// Created by wankai on 2020/7/7.
//

#ifndef IMAGE_RETRIEVAL_IMAGEDATABASE_H
#define IMAGE_RETRIEVAL_IMAGEDATABASE_H

#include "brief.h"

#define DEBUG_INFO 0
#define DEBUG 1
#define DEBUG_INFO_Q 1


struct db_info
{
    int id;
    int index;
    vector<cv::KeyPoint> kps;
    vector<BRIEF::bitset> brf_desc;
};

struct qr_info
{
    int id;
    int index;
    double score;
};

class ImageDatabase {
public:
    ImageDatabase(string voc_path, std::string _pattern_file);

    #ifdef DEBUG
    std::vector<pair<int,vector<string>>> images_DBList;
    std::vector<pair<int,string>> images_QRList;
    void saveImagePath(const int id, const vector<string> &imagesList);
    void addImage(const cv::Mat &image, int set_id, int set_id_index);
    pair<int,int> query(cv::Mat image);
    pair<int, double> query_list(int set_id, const std::vector<cv::Mat>& image_list);
    #else
    void addImage(const cv::Mat &image, int set_id);
    int query(cv::Mat image);
    pair<int, double> query_list(const std::vector<cv::Mat>& image_list);
    #endif
    bool erase(int id);
    void extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors);

    int scene_num;
    BriefDatabase db;

private:
    int counter;
    #ifdef DEBUG
    // std::vector<std::pair<int,int>> imageset_id;
    std::vector<db_info> imageset_id;
    #else
    vector<int> imageset_id;
    #endif
    std::string pattern_file;
    DBoW2::QueryResults ret;
    void blurImage4Brief(const cv::Mat &src, cv::Mat &dst);
    void computeBRIEFPoint(const cv::Mat &image, cv::Mat &image_blur,vector<cv::KeyPoint> &keypoints,
                           vector<BRIEF::bitset> &brief_descriptors);
    int id_cnt;
    int last_id,last_id1;
};


#endif //IMAGE_RETRIEVAL_IMAGEDATABASE_H
