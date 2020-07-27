//
// Created by wankai on 2020/7/16.
//

#ifndef IMAGE_RETRIEVAL_IMDB_SDK_H
#define IMAGE_RETRIEVAL_IMDB_SDK_H

#include <string>
#include <iostream>
#include <vector>

#ifdef __cplusplus
#ifdef WINDOWS
#ifdef API_EXPORT_DLL
#define API_EXPORT extern "C" __declspec(dllexport)
#else
#define API_EXPORT extern "C" __declspec(dllimport)
#endif
#else
#define API_EXPORT extern "C"
#endif
#else
#define API_EXPORT
#endif
struct query_result
{
    int set_id;
    float confidence;
};

API_EXPORT void* initDataBase(std::string voc_path, std::string _pattern_file);
API_EXPORT bool addImage(void* handler, const std::string &img_path, int set_id);
API_EXPORT int query(void* handler, const std::string &img_path);
API_EXPORT query_result query_list(void* handler, const char* pData, int nWidth, int nHeight, int numFrame);
API_EXPORT bool erase(void* handler, int id);
API_EXPORT bool releaseDataBase(void* handler);

#endif //IMAGE_RETRIEVAL_IMDB_SDK_H
