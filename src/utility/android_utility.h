#ifndef _android_jni_utility_H__
#define _android_jni_utility_H__

#include <string>
#include <cmath>
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
using namespace std;


typedef unsigned short ushort;
#define LOG_TAG "JNI"
#define SLAM_LOG_INTERVAL 1

#if 1

#ifdef __ANDROID__
#include<android/log.h>

#if 0
#define LOGI(fmt, args...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, fmt, ##args)
#else
int my_printf( const char* format, ...); 
#define LOGI my_printf
#endif

#define LOGD(fmt, args...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, fmt, ##args)
#define LOGE(fmt, args...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##args)
#else
#ifdef WINDOWS
extern int SAVE_LOG;
#define LOGI(...)
#define LOGD(...) do{if(0!=SAVE_LOG) printf(__VA_ARGS__);}while(0)
#define LOGE(...)
#else
#define LOGI(fmt, args...) do{printf(fmt, ##args);printf("\n");}while(0)
#define LOGD(fmt, args...) do{printf(fmt, ##args);printf("\n");}while(0)
#define LOGE(fmt, args...) do{printf(fmt, ##args);printf("\n");}while(0)
#endif//WINDOWS
#endif

#else
int my_printf( const char* format, ...); 
#define LOGI my_printf
#endif

#endif


