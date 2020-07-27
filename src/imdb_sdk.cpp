//
// Created by wankai on 2020/7/16.
//

#include "ImageDatabase.h"
#include "imdb_sdk.h"


API_EXPORT void* initDataBase(string voc_path, std::string _pattern_file)
{
    ImageDatabase* imdb = new ImageDatabase(voc_path,  _pattern_file);
    imdb->scene_num = SCENE_NUM;
    return imdb;
}

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
        memcpy(image.data, pData + nWidth * nHeight * i, nWidth * nHeight);
        if(DEBUG_INFO) {
            cv::imshow("image", image);
            cv::waitKey(5);
        }
        images.push_back(image);
    }
    pair<int, double> id_query = imdb->query_list(images);
    qr.set_id = id_query.first;
    qr.confidence = id_query.second;
    return  qr;
}

API_EXPORT bool erase(void* handler, int id) {
    ImageDatabase* imdb = (ImageDatabase*)handler;
    imdb->erase(id);
}

API_EXPORT bool releaseDataBase(void* handler){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    delete(imdb);
}

