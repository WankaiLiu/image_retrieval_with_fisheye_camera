//
// Created by wankai on 2020/7/7.
//

#ifndef IMAGE_RETRIEVAL_IMAGEDATABASE_H
#define IMAGE_RETRIEVAL_IMAGEDATABASE_H

#include "brief.h"

#define DEBUG_INFO 0
#define DEBUG 0
#define DEBUG_INFO_Q 1
#define DEBUG_IMG 0
#define MIN_SCORE 0.2
#define MIN_FUNDAMENTAL_THRESHOLD 40.f


struct db_info
{
    int id;
    int index;
    vector<cv::KeyPoint> kps;
    vector<BRIEF::bitset> brf_desc;
};

class ImageDatabase {
public:
    ImageDatabase(string voc_path, std::string _pattern_file);
    void addImage(const cv::Mat &image, int set_id);
    std::vector<pair<int, double>> query_list(const std::vector<cv::Mat>& image_list);
    bool erase(int id);
    bool erase_set(int set_id);
    void extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors);
    int scene_num;
    int get_dbsize();
    BriefDatabase db;
#if DEBUG_IMG
    void addImagePath(const string &img_path);
#endif

private:
    int counter;
    // std::vector<std::pair<int,int>> imageset_id;
    std::vector<db_info> imageset_id;
#if DEBUG_IMG
    std::vector<string> image_path_vec;
#endif
    std::string pattern_file;
    DBoW2::QueryResults ret;
    void blurImage4Brief(const cv::Mat &src, cv::Mat &dst);
    void computeBRIEFPoint(const cv::Mat &image, cv::Mat &image_blur,vector<cv::KeyPoint> &keypoints,
                           vector<BRIEF::bitset> &brief_descriptors);

};


#endif //IMAGE_RETRIEVAL_IMAGEDATABASE_H
