//
// Created by wankai on 2020/7/16.
//

#include "ImageDatabase.h"
#include "imdb_sdk.h"


API_EXPORT void* initDataBase(string voc_path, std::string pattern_file)
{
    ImageDatabase* imdb = new ImageDatabase(voc_path,  pattern_file);
    imdb->scene_num = 1;
    return imdb;
}

void addImage(void* handler, const string &img_path, int set_id){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    if(imdb->scene_num < set_id + 1) imdb->scene_num = set_id + 1;
    cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
    imdb->addImage(image, set_id);
}

query_result query_list(void* handler, const char* pData, int nWidth, int nHeight, int numFrame) {
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
    return true;
}

API_EXPORT bool releaseDataBase(void* handler){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    delete(imdb);
    return true;
}

