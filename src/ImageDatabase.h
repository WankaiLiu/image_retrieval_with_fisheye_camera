//
// Created by wankai on 2020/7/7.
//

#ifndef IMAGE_RETRIEVAL_IMAGEDATABASE_H
#define IMAGE_RETRIEVAL_IMAGEDATABASE_H

#include "brief.h"

#define DEBUG_INFO 0

class ImageDatabase {
public:
    ImageDatabase(string voc_path, std::string _pattern_file);
    void addImage(const cv::Mat &image, int set_id);
    int query(cv::Mat image);
    bool erase(int id);
    void extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors);
private:
    int counter;
    vector<int> imageset_id;
    BriefDatabase db;
    std::string pattern_file;
    DBoW2::QueryResults ret;
    void blurImage4Brief(const cv::Mat &src, cv::Mat &dst);
    void computeBRIEFPoint(const cv::Mat &image, cv::Mat &image_blur,vector<cv::KeyPoint> &keypoints,
                           vector<BRIEF::bitset> &brief_descriptors);
};


#endif //IMAGE_RETRIEVAL_IMAGEDATABASE_H
