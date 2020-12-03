//
// Created by wankai on 2020/7/16.
//

#include "ImageDatabase.h"
#include "imdb_sdk.h"
#define VERTION_DATE 20201117

API_EXPORT void* initDataBase(string voc_path, std::string pattern_file)
{
    ImageDatabase* imdb = new ImageDatabase(voc_path,  pattern_file);
    imdb->scene_num = 1;
    cout << "InitDataBase...**** version date :" << VERTION_DATE << endl;
    return imdb;
}

void addImage(void* handler, const string &img_path, int set_id, const std::string &camera_file_path){
    try {
        ifstream myfile(camera_file_path);
        if(myfile.is_open())  {
            string line;
            getline(myfile, line);
            cout << "camera_file_path..." <<  line << endl;

        }
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

std::vector<query_result> query_list_vec(void* handler, const std::vector<std::string> &img_path_vec,
        const std::string &camera_file_path){
    vector<query_result>  qr_vec;
    ImageDatabase *imdb = (ImageDatabase *) handler;
    vector<cv::Mat> images;
    cout << "start query...**** version date :" << VERTION_DATE << endl;
    try {
        ifstream myfile(camera_file_path);
        if(myfile.is_open())  {
            string line;
            getline(myfile, line);
            cout << "camera_file_path..." <<  line << endl;

        }
        for (size_t i = 0; i < img_path_vec.size(); i++) {
            cv::Mat image = cv::imread(img_path_vec[i], CV_LOAD_IMAGE_UNCHANGED);
            images.push_back(image);
            cout << "Query Image Info: " << img_path_vec[i] << " " << image.channels() << " " <<
            image.size <<  " " << image.type() << endl;
        }
    }
    catch(...)
    {
        cerr << "Query Error in Parsing Image!!!: Please check your query data" << endl;
        return qr_vec;
    }
    try {
        std::vector<pair<int, double>> id_query_list = imdb->query_list(images);
        for(size_t j = 0; j < id_query_list.size(); j++) {
            query_result qr;
            qr.get_id = id_query_list[j].first;
            qr.confidence = id_query_list[j].second;
            qr_vec.push_back(qr);
        }
        return  qr_vec;
    }
    catch(...)
    {
        cerr << "Query Error in db_query!!!: Please check your database. Current db_size = " << endl;
        cerr << "Query Error Info: imageset.size = " <<  imdb->get_dbsize() << endl;
        return qr_vec;
    }
}

query_result query_list(void* handler, const char* pData, int nWidth, int nHeight, int numFrame) {
    query_result qr;
    ImageDatabase *imdb = (ImageDatabase *) handler;
    vector<cv::Mat> images;
    cout << "start query...**** version date :" << VERTION_DATE << endl;
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
        std::vector<pair<int, double>> id_query_list = imdb->query_list(images);
        if(id_query_list.empty()) {
            qr.get_id = -1;
            qr.confidence = 0;
        }
        else {
            qr.get_id = id_query_list[0].first;
            qr.confidence = id_query_list[0].second;
        }
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

