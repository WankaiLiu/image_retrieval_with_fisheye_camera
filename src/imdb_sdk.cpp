//
// Created by wankai on 2020/7/16.
//

#include "imdb_sdk.h"
#include "ImageDatabase.h"


API_EXPORT void* initDataBase(string voc_path, std::string _pattern_file)
{
    ImageDatabase* imdb = new ImageDatabase(voc_path,  _pattern_file);
    return imdb;
}
//#ifndef DEBUG
#if 1
API_EXPORT bool addImage(void* handler, const string &img_path, int set_id){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
    imdb->addImage(image, set_id);
}

API_EXPORT int query(void* handler, const string &img_path){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
    return  imdb->query(image);
}

API_EXPORT query_result query_list(void* handler, const char* pData, int nWidth, int nHeight, int numFrame) {
    query_result qr;
    ImageDatabase* imdb = (ImageDatabase*)handler;
    vector<cv::Mat> images;
    for(int i = 0; i < numFrame; i++) {
        cv::Mat image = cv::Mat(nHeight, nWidth, CV_8UC1);
        memcpy(image.data, pData + nWidth * nHeight, nWidth * nHeight);
//        cv::imshow("image", image);
//        cv::waitKey(5);
        images.push_back(image);
    }
    //cv::Mat image = cv::imread(img_path_vec[0], CV_LOAD_IMAGE_UNCHANGED);
    qr.set_id = imdb->query(images[0]);
    qr.confidence = 1;
    return qr;
}
#else
API_EXPORT bool addImage(const string &img_path, int set_id){
}

API_EXPORT int query(const string &img_path){
    return  -1;
}
API_EXPORT int  query_list(const std::vector<std::string> &img_path_vec) {
    return  -1;
}
#endif
API_EXPORT bool erase(void* handler, int id) {
    ImageDatabase* imdb = (ImageDatabase*)handler;
    imdb->erase(id);
}
API_EXPORT bool releaseDataBase(void* handler){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    delete(imdb);
}

