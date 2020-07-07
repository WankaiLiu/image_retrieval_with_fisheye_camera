#include <thread>
#include "visualization.h"

using namespace std;

namespace slam_interface
{
    extern std::string save_dir;
}

#ifdef VIZ_TRACE
#include "opencv2/viz.hpp"
Eigen::Vector3d viz_P;
Eigen::Quaterniond viz_Q;
unsigned long itercnt;
cv::Point3d point_end;
cv::Point3d point_begin;
cv::Point3d cam_pos;
cv::Point3d cam_y_dir;
cv::Affine3d cam_pose;
cv::viz::Viz3d vis;
cv::viz::WCoordinateSystem world_coor;
cv::viz::WCoordinateSystem camera_coor;
void viz_k(vector<double> &vec)
{
    Eigen::Quaterniond q(vec[6], vec[3], vec[4], vec[5]);
    Eigen::Isometry3d T(q);
    T.pretranslate(Eigen::Vector3d(vec[0], vec[1], vec[2]));
    cv::Affine3d M(
        cv::Affine3d::Mat3(
            T(0, 0), T(0, 1), T(0, 2),
            T(1, 0), T(1, 1), T(1, 2),
            T(2, 0), T(2, 1), T(2, 2)),
        cv::Affine3d::Vec3(
            T.translation()(0, 0),
            T.translation()(1, 0),
            T.translation()(2, 0)));
    std::vector<cv::viz::WLine> lines;
    point_end = cv::Point3d(
        T.translation()(0, 0),
        T.translation()(1, 0),
        T.translation()(2, 0));

    cv::viz::WLine line(point_begin, point_end, cv::viz::Color::green());
    lines.push_back(line);
    for (vector<cv::viz::WLine>::iterator iter = lines.begin(); iter != lines.end(); iter++)
    {
        string id = to_string(itercnt);
        vis.showWidget(id, *iter);
        itercnt++;
    }
    point_begin = point_end;
    vis.setWidgetPose("Camera", M);
    //vis.setWindowSize(cv::Size(640,400));
    vis.spinOnce(1, false);
}
#endif

#ifdef VIZ_OFLOW_MATCH
cv::Mat oflow_img;
vector<cv::Mat> old_imgs;
void oflow_show(const cv::Mat &image,
                               const string &title,
                               const vector<int> &ids,
                               const vector<cv::Point2f> &pts_b, // base points
                               const vector<cv::Point2f> &pts_s, // second points
                               const vector<uchar> &status,
                               int layout)
{
    if(!image.empty())
    {
        cv::Mat color_img;
        cvtColor(image, color_img, cv::COLOR_GRAY2RGB);

        cv::RNG rng(12345);
        static vector<cv::Scalar> colors;
        if(colors.empty())
        {
             for(int i = 0; i < 10000; i++)
             {
                 colors.push_back(cv::Scalar(rng.uniform(0, 255),
                                             rng.uniform(0, 255),
                                             rng.uniform(0, 255)));
             }
        }

        cv::Point2f offset_b(0, 0);
        cv::Point2f offset_s(0, 0);
        if (layout == 1)
        {
            offset_b.y = ROW;
        }
        else if (layout == 2)
        {
            offset_s.x = COL;
        }

        for (int i = 0; i < pts_b.size(); i++)
        {
            int id = ids[i];
            cv::Scalar color = colors[id % 10000];
            cv::Point pos_b = pts_b[i] + offset_b;
            cv::Point pos_s = pts_s[i] + offset_s;

            cv::circle(color_img, pos_b, 3, color, -1);
            if(!status.empty() && status[i])
            {
                cv::circle(color_img, pos_s, 3, color, -1);
                cv::line(color_img, pos_b, pos_s, color, 1.5, 8);
            }

            char buf[10];
            sprintf(buf, "%d", id);
            cv::Point pos = pts_b[i] + offset_b + cv::Point2f(4.0, -5.0);
            if (pts_b[i].y < 20) pos.y += 20;
            if (pts_b[i].x > COL - 30) pos.x -= 50;
            cv::putText(color_img, buf, pos, cv::FONT_HERSHEY_COMPLEX, 0.5, color);
        }

        cv::imshow(title.c_str(), color_img);
        cv::waitKey(1);
    }
}
#endif

#if 0
using namespace ros;
using namespace Eigen;

ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path, pub_loop_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;

ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual, pub_pose_graph;
nav_msgs::Path path, loop_path;
CameraPoseVisualization cameraposevisual(0, 0, 1, 1);
CameraPoseVisualization keyframebasevisual(0.0, 0.0, 1.0, 1.0);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path_no_loop", 1000);
    pub_loop_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("history_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<geometry_msgs::PoseStamped>("camera_pose", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_pose_graph = n.advertise<visualization_msgs::MarkerArray>("pose_graph", 1000);

    cameraposevisual.setScale(0.2);
    cameraposevisual.setLineWidth(0.01);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header)
{
    Eigen::Quaterniond quadrotor_Q = Q ;

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);
}

void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    ROS_INFO_STREAM("position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[WINDOW_SIZE].transpose());
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        //ROS_DEBUG("calibration result for camera %d", i);
        ROS_DEBUG_STREAM("extirnsic tic: " << estimator.tic[i].transpose());
        ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());
        if (ESTIMATE_EXTRINSIC)
        {
            cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
            Eigen::Matrix3d eigen_R;
            Eigen::Vector3d eigen_T;
            eigen_R = estimator.ric[i];
            eigen_T = estimator.tic[i];
            cv::Mat cv_R, cv_T;
            cv::eigen2cv(eigen_R, cv_R);
            cv::eigen2cv(eigen_T, cv_T);
            fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
            fs.release();
        }
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_DEBUG("sum of path %f", sum_of_path);
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                Eigen::Matrix3d loop_correct_r)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = loop_correct_r * estimator.key_poses[i] + loop_correct_t;
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                   Eigen::Matrix3d loop_correct_r)
{
    int idx2 = WINDOW_SIZE - 1;
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        int i = idx2;
        geometry_msgs::PoseStamped camera_pose;
        camera_pose.header = header;
        camera_pose.header.frame_id = std::to_string(estimator.Headers[i].stamp.toNSec());
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);
        P = (loop_correct_r * estimator.Ps[i] + loop_correct_t) + (loop_correct_r * estimator.Rs[i]) * estimator.tic[0];
        R = Quaterniond((loop_correct_r * estimator.Rs[i]) * estimator.ric[0]);
        camera_pose.pose.position.x = P.x();
        camera_pose.pose.position.y = P.y();
        camera_pose.pose.position.z = P.z();
        camera_pose.pose.orientation.w = R.w();
        camera_pose.pose.orientation.x = R.x();
        camera_pose.pose.orientation.y = R.y();
        camera_pose.pose.orientation.z = R.z();

        pub_camera_pose.publish(camera_pose);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        camera_pose.header.frame_id = "world";
        cameraposevisual.publish_by(pub_camera_pose_visual, camera_pose.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                   Eigen::Matrix3d loop_correct_r)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                              + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud, loop_margin_cloud;
    margin_cloud.header = header;
    loop_margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
            && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = loop_correct_r * estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                             + loop_correct_r * estimator.Ps[imu_i] + loop_correct_t;

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}

void pubPoseGraph(CameraPoseVisualization* posegraph, const std_msgs::Header &header)
{
    posegraph->publish_by(pub_pose_graph, header);
    
}

void updateLoopPath(nav_msgs::Path _loop_path)
{
    loop_path = _loop_path;
}

void pubTF(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                   Eigen::Matrix3d loop_correct_r)
{
    if( estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = loop_correct_r * estimator.Ps[WINDOW_SIZE] + loop_correct_t;
    correct_q = loop_correct_r * estimator.Rs[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));
}
void pubOdometry(const Estimator &estimator, const std_msgs::Header &header, Eigen::Vector3d loop_correct_t,
                Eigen::Matrix3d loop_correct_r)
{
    return ;//sindo
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
  /*      nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = Quaterniond(estimator.Rs[WINDOW_SIZE]).x();
        odometry.pose.pose.orientation.y = Quaterniond(estimator.Rs[WINDOW_SIZE]).y();
        odometry.pose.pose.orientation.z = Quaterniond(estimator.Rs[WINDOW_SIZE]).z();
        odometry.pose.pose.orientation.w = Quaterniond(estimator.Rs[WINDOW_SIZE]).w();
        
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);*/

        Vector3d correct_t;
        Vector3d correct_v;
        Quaterniond correct_q;
        correct_t = loop_correct_r * estimator.Ps[WINDOW_SIZE] + loop_correct_t;
        correct_q = loop_correct_r * estimator.Rs[WINDOW_SIZE];
        correct_v = loop_correct_r * estimator.Vs[WINDOW_SIZE];
 /*       odometry.pose.pose.position.x = correct_t.x();
        odometry.pose.pose.position.y = correct_t.y();
        odometry.pose.pose.position.z = correct_t.z();
        odometry.pose.pose.orientation.x = correct_q.x();
        odometry.pose.pose.orientation.y = correct_q.y();
        odometry.pose.pose.orientation.z = correct_q.z();
        odometry.pose.pose.orientation.w = correct_q.w();
        odometry.twist.twist.linear.x = correct_v(0);
        odometry.twist.twist.linear.y = correct_v(1);
        odometry.twist.twist.linear.z = correct_v(2);
        pub_odometry.publish(odometry);

        pose_stamped.pose = odometry.pose.pose;
        loop_path.header = header;
        loop_path.header.frame_id = "world";
        loop_path.poses.push_back(pose_stamped);
        pub_loop_path.publish(loop_path);
*/
        // write result to file
        ofstream foutC(VINS_RESULT_PATH, ios::app);
        foutC.setf(ios::fixed, ios::floatfield);
        foutC.precision(0);
        foutC << header.stamp.toSec() * 1e9 << ",";
        foutC.precision(5);
        foutC << correct_t.x() << ","
              << correct_t.y() << ","
              << correct_t.z() << ","
              << correct_q.w() << ","
              << correct_q.x() << ","
              << correct_q.y() << ","
              << correct_q.z() << ","
              << correct_v(0) << ","
              << correct_v(1) << ","
              << correct_v(2) << "," << endl;
        foutC.close();
    }
}
#endif




//socket related functions
#ifndef WINDOWS
namespace SOCKET_SEND_GLOBAL
{
    SOCKET_INFO socket_info;
    std::mutex m_socket_info;
};

using namespace SOCKET_SEND_GLOBAL;

void local_signal_process()
{
    int sockfd = socket(PF_LOCAL, SOCK_DGRAM, 0);
    if(sockfd < 0)
    {
        LOGD("init socket failed");
        return;
    }

    const char* sockname = "slam_socket";

    sockaddr_un serveraddr;
    sockaddr_in clientaddr;
    memset(&serveraddr, 0, sizeof(serveraddr));
    serveraddr.sun_family = PF_LOCAL;
    serveraddr.sun_path[0] = 0;
    strcpy(serveraddr.sun_path+1, sockname);

    char *sun_path = serveraddr.sun_path;
    int socklen = (offsetof(sockaddr_un, sun_path) ) + strlen(sockname) + 1;

    if(bind(sockfd, (struct sockaddr *)&serveraddr, socklen) < 0)
    {
        LOGD("bind fail\n");
    }
    LOGD("init socket info ok\n");
    char* message = new char[200];

    while (1)
    {
        LOGD("socket: waiting for command...\n");

        int client_len = sizeof(clientaddr);
        // block here until some info received
        int recv_len = recvfrom(sockfd, message, sizeof(message), 0,
                                (sockaddr*)&serveraddr,
                                (socklen_t *)&socklen);
        if (recv_len < 0)
        {
            LOGD("Recive Error");
            continue;
        }

        if (message[0] == '#' && message[1] == '#')
        {
            char command = message[2];
            char value   = message[3];
            LOGD("socket: recieve command(%d, %d)\n\n", command, value);

            if(command == 1) // DEBUG_SAVE_FILE
            {
                if(value == 0) DEBUG_SAVE_FILE = 0; // dont save data
                if(value == 1) DEBUG_SAVE_FILE = 1; // save 6dof data
                if(value == 2) {                    // save raw  data
                    DEBUG_SAVE_FILE = 2;
                    save_data.reset(slam_interface::save_dir);
                }
            }
            if(command == 2) // data_check
            {
                if(value == 0) data_check = 0; // dont check data
                if(value == 1) data_check = 1; // show stat  info
                if(value == 2) data_check = 2; // show more  info
                check_data.setImageCount(300);
                check_data.setMode(data_check);
                check_data.reset(400,30);
            }
        }
    }

    delete []message;
}

void start_sever_listener()
{
    LOGD("socket: start_sever_listener");
    std::thread thread_signal = thread(&local_signal_process);
    //thread_signal.detach();
    thread_signal.detach();
}

bool init_socket_info()
{
    LOGD("init socket info start\n");
    static int init_ok = false;
    if(init_ok)return true;

    m_socket_info.lock();
    socket_info.sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if(socket_info.sockfd < 0)
    {
        m_socket_info.unlock();
        LOGE("init socket failed");
        return false;
    }
    int opt = 1;
    setsockopt(socket_info.sockfd, SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));

    memset(&socket_info.serveraddr, 0, sizeof(socket_info.serveraddr));
    socket_info.serveraddr.sin_family = AF_INET;
    socket_info.serveraddr.sin_port = htons(11166); //port
    socket_info.serveraddr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    //inet_ntop(AF_INET,  (struct sockaddr*)&serveraddr.sin_addr.s_addr, argv[1], sizeof(argv[1]));

    LOGD("init socket info ok\n");
    init_ok = true;

    m_socket_info.unlock();

    return true;
}

void send_socket_info(int command_code)
{
    char send_buf[2];
    send_buf[0] = '#';

    send_buf[1] = ' ';

    if(command_code == 9){
        send_buf[1] = '9';

        size_t size = 2;

        m_socket_info.lock();
        int ret = sendto(socket_info.sockfd, send_buf, size, 0, (struct sockaddr *)&socket_info.serveraddr, sizeof(socket_info.serveraddr));
        ret = sendto(socket_info.sockfd, send_buf, size, 0, (struct sockaddr *)&socket_info.serveraddr, sizeof(socket_info.serveraddr));
        ret = sendto(socket_info.sockfd, send_buf, size, 0, (struct sockaddr *)&socket_info.serveraddr, sizeof(socket_info.serveraddr));
        m_socket_info.unlock();

        if(ret < 0)
        {
            LOGD("reset broadcast fail\n");
        }
        else
        {
            LOGD("reset broadcastn success!\n");
        }
    } 
}

void send_socket_info(const Pose_6dof &output_pose)
{
    char send_buf[100];
    send_buf[0] = '#';
    send_buf[1] = '0';

    memcpy(send_buf+2, &output_pose.dTime, 8 * sizeof(double));
    
    //sixdof[0] = output_pose.dTime;
    //sixdof[1] = output_pose.dX[0];
    //sixdof[2] = output_pose.dX[1];
    //sixdof[3] = output_pose.dX[2];
    
    //sixdof[4] = output_pose.dQuaternion[0];
    //sixdof[5] = output_pose.dQuaternion[1];
    //sixdof[6] = output_pose.dQuaternion[2];
    //sixdof[7] = output_pose.dQuaternion[3];


    size_t size = 2 + 8 * sizeof(double);

    m_socket_info.lock();
    int ret = sendto(socket_info.sockfd, send_buf, size, 0, (struct sockaddr *)&socket_info.serveraddr, sizeof(socket_info.serveraddr));
    m_socket_info.unlock();

    if(ret < 0)
    {
        LOGD("broadcast fail\n");
    }
    else
    {
        LOGD("broadcastn success!\n");
    }
}


void send_socket_info(const vector<Point3D> &output_points)
{
    if(output_points.size() == 0) 
    {
        return;
    }
    char *send_buf = new char[output_points.size()*sizeof(Point3D) + 100];
    send_buf[0] = '#';
    send_buf[1] = '1';


    for(int i = 0; i < output_points.size(); ++i)
    {
        memcpy(send_buf+2+3*sizeof(float)*i, &(output_points[i].x), 3*sizeof(float));
    }

    size_t size = 2 + 3 * sizeof(float) * output_points.size();

    m_socket_info.lock();
    int ret = sendto(socket_info.sockfd, send_buf, size, 0, (struct sockaddr *)&socket_info.serveraddr, sizeof(socket_info.serveraddr));
    m_socket_info.unlock();

    delete []send_buf;

    if(ret < 0)
    {
        LOGD("point cloud broadcast fail\n");
    }
    else
    {
        LOGD("point cloud broadcastn success!\n");
    }
}



void send_socket_info(const vector<uchar> &encode)
{
    if(encode.size() == 0 || encode.size() > (40000)) {
        return;
    }
    //static char send_buf[320*240+2];
    char *send_buf = new char[2+encode.size()];
    send_buf[0] = '#';
    send_buf[1] = '2';

    memcpy(send_buf+2, &encode[0], encode.size());

    size_t size = 2 + encode.size();
    m_socket_info.lock();
    int ret = sendto(socket_info.sockfd, send_buf, size, 0, (struct sockaddr *)&socket_info.serveraddr, sizeof(socket_info.serveraddr));
    m_socket_info.unlock();

    delete []send_buf;

    if(ret < 0)
    {
        LOGD("Send image broadcast fail size=%zu\n", size);
    }
    else
    {
        LOGD("Send image broadcastn success! size = %zu\n", size);
    }
}
#endif