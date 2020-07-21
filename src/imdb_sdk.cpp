//
// Created by wankai on 2020/7/16.
//

#include "imdb_sdk.h"
#include "ImageDatabase.h"

ImageDatabase* imdb;
API_EXPORT bool initDataBase(string voc_path, std::string _pattern_file)
{
    imdb = new ImageDatabase(voc_path,  _pattern_file);
    return true;
}
API_EXPORT bool addImage(const string &img_path, int set_id){
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
    imdb->addImage(image, set_id);
}

API_EXPORT int query(const string &img_path){
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
    return  imdb->query(image);
}
API_EXPORT int  query_list(const std::vector<std::string> &img_path_vec) {
    cv::Mat image = cv::imread(img_path_vec[0], CV_LOAD_IMAGE_UNCHANGED);
    return  imdb->query(image);
}
API_EXPORT bool erase(int id) {
    imdb->erase(id);
}
API_EXPORT bool releaseDataBase(){
    delete(imdb);
}

