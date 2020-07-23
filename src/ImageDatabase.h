//
// Created by wankai on 2020/7/7.
//

#ifndef IMAGE_RETRIEVAL_IMAGEDATABASE_H
#define IMAGE_RETRIEVAL_IMAGEDATABASE_H

#include "brief.h"

#define DEBUG_INFO 0
//#define DEBUG

class ImageDatabase {
public:
    ImageDatabase(string voc_path, std::string _pattern_file);
    #ifndef DEBUG
    void addImage(const cv::Mat &image, int set_id);
    int query(cv::Mat image);
    #else
    BriefDatabase db;
    void addImage(const cv::Mat &image, int set_id, int set_id_index, int create_db);
    pair<int,int> query(cv::Mat image);
    #endif
    bool erase(int id);
    void extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors);
private:
    int counter;
    #ifndef DEBUG
    vector<int> imageset_id;
    BriefDatabase db;
    #else
    vector<pair<int,int>>  imageset_id;
    #endif
    std::string pattern_file;
    DBoW2::QueryResults ret;
    void blurImage4Brief(const cv::Mat &src, cv::Mat &dst);
    void computeBRIEFPoint(const cv::Mat &image, cv::Mat &image_blur,vector<cv::KeyPoint> &keypoints,
                           vector<BRIEF::bitset> &brief_descriptors);
};


#endif //IMAGE_RETRIEVAL_IMAGEDATABASE_H
