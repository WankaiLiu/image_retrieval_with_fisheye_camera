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

API_EXPORT bool initDataBase(std::string voc_path, std::string _pattern_file);
API_EXPORT bool addImage(const std::string &img_path, int set_id);
API_EXPORT int query(const std::string &img_path);
API_EXPORT int query_list(const std::vector<std::string> &img_path_vec);
API_EXPORT bool erase(int id);
API_EXPORT bool releaseDataBase();

#endif //IMAGE_RETRIEVAL_IMDB_SDK_H
