#pragma once

#include "parameters.h"
#include "../vio/slam_sdk.h"
#include <vector>
#include <mutex>

#ifdef VIZ_OFLOW_MATCH
    extern cv::Mat oflow_img;
    extern vector<cv::Mat> old_imgs;
    void oflow_show(const cv::Mat &image,
                    const string &title,
                    const vector<int> &ids,
                    const vector<cv::Point2f> &pts_b, // base points
                    const vector<cv::Point2f> &pts_s, // second points
                    const vector<uchar> &status,
                    int layout);
#endif

#ifdef VIZ_TRACE
    extern Eigen::Vector3d viz_P;
    extern Eigen::Quaterniond viz_Q;
    extern unsigned long itercnt;
    extern cv::Point3d point_end;
    extern cv::Point3d point_begin;
    extern cv::Point3d cam_pos;
    extern cv::Point3d cam_y_dir;
    extern cv::Affine3d cam_pose;
    extern cv::viz::Viz3d vis;
    extern cv::viz::WCoordinateSystem world_coor;
    extern cv::viz::WCoordinateSystem camera_coor;
    void viz_k(vector<double> &vec);
#endif

//#include <ros/ros.h>
#if 0
#include <eigen3/Eigen/Dense>


#include "../../include/Vector3.h"
#include "../../include/Quaternion.h"
#include <fstream>

#include "estimator.h"
void pubOdometry(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                Eigen::Matrix3d loop_correct_r);

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "../estimator.h"
#include "../parameters.h"
#include <fstream>

extern ros::Publisher pub_odometry;
extern ros::Publisher pub_path, pub_pose;
extern ros::Publisher pub_cloud, pub_map;
extern ros::Publisher pub_key_poses;

extern ros::Publisher pub_ref_pose, pub_cur_pose;

extern ros::Publisher pub_key;

extern nav_msgs::Path path;

extern ros::Publisher pub_pose_graph;

void registerPub(ros::NodeHandle &n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                Eigen::Matrix3d loop_correct_r);

void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
				Eigen::Matrix3d loop_correct_r);

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                   Eigen::Matrix3d loop_correct_r);

void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                   Eigen::Matrix3d loop_correct_r);

void pubPoseGraph(CameraPoseVisualization* posegraph, const std_msgs::Header &header);

void updateLoopPath(nav_msgs::Path _loop_path);

void pubTF(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                   Eigen::Matrix3d loop_correct_r);
#endif

#ifndef WINDOWS
//socket functions
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/un.h>
struct SOCKET_INFO
{
    int sockfd;
    sockaddr_in serveraddr;
};

int prepareDebugSaveDir(const std::string path);

bool init_socket_info();
void send_socket_info(int command_code);
void send_socket_info(const Pose_6dof &output_pose);
void send_socket_info(const Point3D &output_point);
void send_socket_info(const std::vector<Point3D> &output_points);
void send_socket_info(const std::vector<uchar> &encode);
#endif
void start_sever_listener();
