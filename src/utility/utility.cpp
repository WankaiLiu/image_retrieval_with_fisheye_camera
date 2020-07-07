#include "utility.h"
#include "android_utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}


#include "slam_sdk.h"
#include <mutex>
namespace slam_interface
{
extern NoticeCallback *p_notice_callback;
extern std::mutex  mtx_notice_callback;
extern ScanProgressCallback *p_scan_progress_callback;
extern std::mutex  mtx_scan_progress_callback;
}

void user_notify(int id, const char* str)
{
    slam_interface::mtx_notice_callback.lock();
    if(slam_interface::p_notice_callback)
    {
        slam_interface::p_notice_callback(id, str);
    }
    slam_interface::mtx_notice_callback.unlock();
    LOGD("%s(%d, %s)", __func__, id, str);
}

void user_scan_progress_notify(int progress)
{
    slam_interface::mtx_scan_progress_callback.lock();
    if(slam_interface::p_scan_progress_callback){
        slam_interface::p_scan_progress_callback(progress);
    }
    slam_interface::mtx_scan_progress_callback.unlock();
}
 
#ifdef __ANDROID__
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
//#include <stdio.h>
//#include <stdlib.h>
#include <unistd.h>

char const *  CPUPATH = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";
char cpu_current_mode[][32] = {"performance", "conservative", "ondemand", "userspace", "powersave", "schedutil"};
char cpu_mode[32] = {0};
 
static int s_pthread_exit = 0;
static pthread_t s_pthread_t;
#endif
static void sysfs_write(const char* path, const int mode)
{
#ifdef __ANDROID__
    char buffer[32] = {0};
    int fd = open(path, O_RDWR, S_IRWXO|S_IRWXG|S_IRWXU);
    if (fd < 0)
    {
        strerror_r(errno, buffer, sizeof(buffer));
        LOGE("Error opening %s: %s\n", CPUPATH, buffer);
        return;
    }
    read(fd, buffer, sizeof(buffer));
    //LOGD("****** current mode is %s", buffer);
    buffer[strlen(buffer)-1]=0; //去除结束符
    if(strcmp(buffer, cpu_current_mode[0]) != 0)
    {
        int len=0;
        len = write(fd, cpu_current_mode[mode], sizeof(cpu_current_mode[mode]));
        if (len < 0)
        {
            char buffer[32] = {0};
            strerror_r(errno, buffer, sizeof(buffer));
            LOGE("Error writing to %s: %s\n", CPUPATH, buffer);
        }
        LOGD("****** set cpu mode is %s", cpu_current_mode[mode]);
    }

    close(fd); 
#endif 
}
 
// NOTE: There are some dummy open and close function call, 
// this is very important, if not, read and write will failed
static void *cpu_status_control(void *p)
{
#ifdef __ANDROID__
    char buffer[32] = {0};
    int fd = open(CPUPATH, O_RDWR, S_IRWXO|S_IRWXG|S_IRWXU);
    if (fd < 0)
    {
        strerror_r(errno, buffer, sizeof(buffer));
        LOGE("Error opening %s: %s\n", CPUPATH, buffer);
        return NULL;
    }
 
    //获得初始cpu默认模式
    read(fd, buffer, sizeof(buffer));
    buffer[strlen(buffer)-1]=0; //去除结束符
    if(strlen(cpu_mode) == 0 && strcmp(buffer, cpu_current_mode[0]) != 0)
    {
        strcpy(cpu_mode, buffer);
    }

    close(fd);

    while(!s_pthread_exit)
    {
        sysfs_write(CPUPATH, 0);
        usleep(3000000);
    }

    fd = open(CPUPATH, O_RDWR, S_IRWXO|S_IRWXG|S_IRWXU);
    if (fd < 0)
    {
        strerror_r(errno, buffer, sizeof(buffer));
        LOGE("Error opening %s: %s\n", CPUPATH, buffer);
        return NULL;
    }

    //线程退出，恢复默认cpu mode
    int len = write(fd, cpu_mode, sizeof(cpu_mode));
    if (len < 0)
    {
        strerror_r(errno, buffer, sizeof(buffer));
        LOGE("Error writing to %s: %s\n", CPUPATH, buffer);
    }

    close(fd);
    return NULL;
#endif
}
 
/**
 * 开启cpu 设置线程
 */
void start_cpu_status_control_process(void)
{
#ifdef __ANDROID__
    s_pthread_exit = 0;
    pthread_create(&s_pthread_t, NULL, cpu_status_control, NULL);
#endif
}
 
/**
 * 退出cpu 设置线程
 */
void stop_cpu_status_control_process(void)
{
#ifdef __ANDROID__
    int ret;
    s_pthread_exit = 1;
    ret = pthread_kill(s_pthread_t, 0);
    if(ret == ESRCH)
    {
        LOGD("pthread not found\n");
    }
    else if(ret == EINVAL)
    {
        LOGD("send an illegal signal\n");
    }
    else
    {
        LOGD("pthread still alive\n");
    }
#endif
}
