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
    try {
        ImageDatabase* imdb = (ImageDatabase*)handler;
        if(imdb->scene_num < set_id + 1) imdb->scene_num = set_id + 1;
        cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
        imdb->addImage(image, set_id);
#if DEBUG_IMG
        imdb->addImagePath(img_path);
#endif
    }
    catch(...)
    {
        cerr << "addImage Error!!!: Please check your file" << img_path << endl;
    }


}

query_result query_list(void* handler, const char* pData, int nWidth, int nHeight, int numFrame) {
    query_result qr;
    ImageDatabase *imdb = (ImageDatabase *) handler;
    vector<cv::Mat> images;
    try {
        for (int i = 0; i < numFrame; i++) {
            cv::Mat image = cv::Mat(nHeight, nWidth, CV_8UC1);
            memcpy(image.data, pData + nWidth * nHeight * i, nWidth * nHeight);
            if (DEBUG_INFO) {
                cv::imshow("image", image);
                cv::waitKey(5);
            }
            images.push_back(image);
        }
    }
    catch(...)
    {
        cerr << "Query Error in Parsing Image!!!: Please check your query data" << endl;
        qr.get_id = -2;
        return qr;
    }
    try {
        pair<int, double> id_query = imdb->query_list(images);
        qr.get_id = id_query.first;
        qr.confidence = id_query.second;
        return  qr;
    }
    catch(...)
    {
        cerr << "Query Error in db_query!!!: Please check your database. Current db_size = " << endl;
        cerr << "Query Error Info: imageset.size = " <<  imdb->get_dbsize() << endl;
        qr.get_id = -2;
        return qr;
    }


}


API_EXPORT bool erase(void* handler, int id) {
    ImageDatabase* imdb = (ImageDatabase*)handler;
    return imdb->erase(id);
}

API_EXPORT bool erase_set(void* handler, int set_id){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    return imdb->erase_set(set_id);
}

API_EXPORT bool releaseDataBase(void* handler){
    ImageDatabase* imdb = (ImageDatabase*)handler;
    delete(imdb);
    return true;
}

