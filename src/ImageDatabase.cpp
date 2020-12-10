//
// Created by wankai on 2020/7/7.
//

#include "ImageDatabase.h"
#include "undistorter.h"
#include <unordered_set>
#include "tic_toc.h"
#include <thread>

#include <opencv2/features2d/features2d.hpp>

ImageDatabase::ImageDatabase(string voc_path, std::string _pattern_file){
    BriefVocabulary* voc = new BriefVocabulary(voc_path);
    this->db.setVocabulary(*voc, true, 0);
    pattern_file = _pattern_file;
    delete voc;
    counter = 0;
    scene_num = 1;
}

void ImageDatabase::blurImage4Brief(const cv::Mat &src, cv::Mat &dst){
    if (!src.empty())
    {
        const float sigma = 2.f;
        const cv::Size ksize(9, 9);
        cv::GaussianBlur(src, dst, ksize, sigma, sigma);
    }
}

void ImageDatabase::computeBRIEFPoint(const cv::Mat &image, cv::Mat &image_blur,vector<cv::KeyPoint> &keypoints,
                       vector<BRIEF::bitset> &brief_descriptors)
{
    static BriefExtractor m_extractor;
    m_extractor.init(pattern_file.c_str());
    int fast_th = 20; // corner detector response threshold
    //cv::Mat tmp_img;
    //cv::GaussianBlur(image, tmp_img, cv::Size(5, 5), 1.2, 0, cv::BORDER_DEFAULT);

    vector<cv::Point2f> tmp_pts;
    int MAX_KEYFRAME_KEYPOINTS = 800;
    cv::goodFeaturesToTrack(image, tmp_pts, MAX_KEYFRAME_KEYPOINTS, 0.01, 3);
    for(int i = 0; i < (int)tmp_pts.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = tmp_pts[i];
        keypoints.push_back(key);
    }
    if(DEBUG_INFO) LOGD("FAST keypoints size = %zu, fast_th = %d", keypoints.size(), fast_th);
    m_extractor(image_blur, keypoints, brief_descriptors);
}


void ImageDatabase::addImage(const cv::Mat &image, int set_id) {
    cv::Mat image_blur;
    blurImage4Brief(image, image_blur);
    vector<cv::KeyPoint> keypoints;
    vector<BRIEF::bitset> brief_descriptors;
    computeBRIEFPoint(image,image_blur,keypoints,brief_descriptors);
    db.add(brief_descriptors);
    // imageset_id.push_back(make_pair(set_id,set_id_index));
    db_info db_info;
    db_info.id = set_id;
//    db_info.index = images_DBList[set_id].size();
    db_info.kps = keypoints;
    db_info.brf_desc = brief_descriptors;
    imageset_id.push_back(db_info);
}


bool ImageDatabase::erase(int id) {
    if(id > imageset_id.size()) {
        return false;
    }
    DBoW2::EntryId entryId = id;
    db.delete_entry(entryId);
    imageset_id.erase(imageset_id.begin() + id);
#if DEBUG_IMG
    image_path_vec.erase(image_path_vec.begin() + id);
#endif
    return true;
};
int ImageDatabase::get_dbsize() {
    return imageset_id.size();
}
bool ImageDatabase::erase_set(int set_id){
    bool find_set = false;
    if(imageset_id.empty()) return false;
    for(int i = imageset_id.size() - 1; i >= 0; i--) {
        if(imageset_id[i].id == set_id) {
            this->erase(i);
            find_set = true;
        }
    }
    return find_set;
};
int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    int dis = 0;

    BRIEF::bitset xor_of_bitset = a ^ b;
    dis= xor_of_bitset.count();
    return dis;
}
bool searchInAera(const BRIEF::bitset window_descriptor,
             const std::vector<BRIEF::bitset> &descriptors_old,
             const std::vector<cv::KeyPoint> &keypoints_old,
             const std::vector<cv::KeyPoint> &keypoints_old_norm,
             cv::Point2f &best_match,
             cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //LOGD("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
        best_match = keypoints_old[bestIndex].pt;
        best_match_norm = keypoints_old_norm[bestIndex].pt;
        return true;
    }
    else
    {
        return false;
    }
}
void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                      std::vector<cv::Point2f> &matched_2d_old_norm,
                      std::vector<uchar> &status,
                      const std::vector<BRIEF::bitset> &descriptors_old,
                      const std::vector<BRIEF::bitset> &window_brief_descriptors,
                      const std::vector<cv::KeyPoint> &keypoints_old,
                      const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
        {
            status.push_back(1);
        }
        else
        {
            status.push_back(0);
        }
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }

}
template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
    {
        if (status[i])
        {
            v[j++] = v[i];
        }
    }
    v.resize(j);
}
void concatImageAndDraw(const cv::Mat& cur_image, vector<cv::Point2f>& matched_points_cur,
                        const cv::Mat& old_image, vector<cv::Point2f>& matched_points_old,
                        vector<cv::Scalar>& colors, string path, bool draw_line)
{
//    cout << "call concatImageAndDraw " << path << endl;
    cv::Mat image_out_tmp,image_out;
    cv::hconcat(cur_image, old_image, image_out_tmp);
    cvtColor(image_out_tmp, image_out, CV_GRAY2BGR);
    for(int i = 0; i < static_cast<int>(matched_points_old.size()); ++i)
    {
        cv::Point2f old_pt = matched_points_old[i];
        old_pt.x += cur_image.cols;
        cv::Scalar& color = colors[i];
        cv::circle(image_out, old_pt, 5, color);
        //cv::circle(image_out, matched_points_cur[i], 5, color);

        if(draw_line)
        {
            cv::line(image_out, matched_points_cur[i], old_pt, color, 1, 8, 0);
        }
//        if(DEBUG_INFO_Q) cout << endl;
    }
    cv::imshow(path, image_out);
    cv::waitKey(190);

}
void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
    int n = (int)matched_2d_cur_norm.size();
    for (int i = 0; i < n; i++)
    {
        status.push_back(0);
    }
    if (n >= 8)
    {
        cv::findFundamentalMat(matched_2d_cur_norm, matched_2d_old_norm, cv::FM_RANSAC, 2.0 / 290, 0.9, status);
    }
}

//Rescale the score into [0 , 1]
double RescaleScore(const double score) {
    if(score < 0.5) { return score;}
    else if(score < 1) {return 0.5 + (score - 0.5)*0.5;}
    else if(score < 3.5) {return 0.75 + (score - 1)*0.1;}
    else {return 1;}
}

std::vector<pair<int, double>> ImageDatabase::query_list(const std::vector<cv::Mat>& image_list){//根据这一个list中的图片直接在当次判断出当前场景ID
    cout << "start  ImageDatabase::query_list: Mat Size = "  << image_list.size() << endl;
    std::vector<pair<int, double>> ret_vec;
    if(scene_num < 1) {
        cerr << "Please make sure the number of scen is larger than 2" << endl;
        return ret_vec;
    }
    int list_size=image_list.size();
    int high_rate_cnt=0;
    std::map<int,unordered_set<int>> result; //set_id, image_id
    int vote_array[list_size][scene_num];
    int vote_array_fun[list_size][scene_num];
    vector<float> vote_array_total(scene_num,0);
    int count_result[4][scene_num];
    int count_result_fun[4][scene_num];
    for (int i = 0; i < list_size; i++) {
        for (int j = 0; j < scene_num; j++) {
            vote_array[i][j] = 0;
            vote_array_fun[i][j] = 0;
            vote_array_total[j] = 0;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < scene_num; j++) {
            count_result[i][j] = 0;
            count_result_fun[i][j] = 0;
        }
    }
    //Fundamental Ransac to filter outliers
    vector<vector<cv::KeyPoint>> kps_list_query;
    vector<vector<BRIEF::bitset>> brf_list_query;
    Eigen::Vector2d focalLength(290.03672785595984, 289.70361075706387);
    Eigen::Vector2d principalPoint(323.1621621450474, 197.5696586534049);
    Eigen::Vector2i resolution(image_list[0].cols, image_list[0].rows);
    Eigen::Vector4d distCoeffs_RadTan(-0.01847757657533866, 0.0575172937703169, -0.06496298696673658, 0.02593645307703224);
    undistorter::PinholeGeometry camera(focalLength, principalPoint, resolution, undistorter::EquidistantDistortion::create(distCoeffs_RadTan));
    double alpha = 1.0, //alphe=0.0: all pixels valid, alpha=1.0: no pixels lost
            scale = 2.0;
    int interpolation = cv::INTER_LINEAR;
    undistorter::PinholeUndistorter undistorter(camera, alpha, scale, interpolation);
    Eigen::Matrix3d cameraMatrix = camera.getCameraMatrix();
    double cu = cameraMatrix(0, 2), cv = cameraMatrix(1, 2);
    double fu = cameraMatrix(0, 0), fv = cameraMatrix(1, 1);

    for (size_t i = 0; i < list_size; i++) {//query
        cv::Mat image_blur;
        blurImage4Brief(image_list[i], image_blur);
        vector<cv::KeyPoint> keypoints_query;
        vector<BRIEF::bitset> brief_descriptors_query;
        computeBRIEFPoint(image_list[i], image_blur, keypoints_query, brief_descriptors_query);
        db.query(brief_descriptors_query, ret, 4, imageset_id.size());
        kps_list_query.push_back(keypoints_query);
        brf_list_query.push_back(brief_descriptors_query);
        vector<pair<pair<int, int>, double>> ret4;
        //wankai:test by Fundamental Ransac
        vector<uchar> status;
        vector<cv::Point2f> matched_2d_cur, matched_2d_cur_base, matched_2d_old;
        vector<cv::Point2f> matched_2d_cur_norm, matched_2d_cur_norm_base, matched_2d_old_norm;
        std::vector<cv::KeyPoint> keypoints_query_norm;
        for (size_t kpt_id = 0; kpt_id < keypoints_query.size(); kpt_id++) {
            cv::Point2f pt2f(keypoints_query[kpt_id].pt.x, keypoints_query[kpt_id].pt.y);
            matched_2d_cur_base.push_back(pt2f);
            Eigen::Vector2d point(keypoints_query[kpt_id].pt.x, keypoints_query[kpt_id].pt.y);
            point(0) = (point(0) - cu) / fu;
            point(1) = (point(1) - cv) / fv;
            camera.distortion->undistort(point);
            cv::KeyPoint pts = keypoints_query[kpt_id];
            pts.pt.x = point(0);
            pts.pt.y = point(1);
            cv::Point2f pt2f_norm(point(0), point(1));
            matched_2d_cur_norm_base.push_back(pt2f_norm);
        }
        if (ret.size() >= 1) {
            for (int j = 0; j < ret.size(); j++) {
                db_info dbInfo = imageset_id[ret[j].Id];
                Eigen::Matrix3d cameraMatrix = camera.getCameraMatrix();
                std::vector<cv::KeyPoint> keypoints_db;
                std::vector<cv::KeyPoint> keypoints_db_norm;
                keypoints_db.clear();
                keypoints_db_norm.clear();
                matched_2d_old.clear();
                matched_2d_old_norm.clear();
                matched_2d_cur = matched_2d_cur_base;
                matched_2d_cur_norm = matched_2d_cur_norm_base;
                for (size_t kpt_id = 0; kpt_id < dbInfo.kps.size(); kpt_id++) {
                    Eigen::Vector2d point(dbInfo.kps[kpt_id].pt.x, dbInfo.kps[kpt_id].pt.y);
                    point(0) = (point(0) - cu) / fu;
                    point(1) = (point(1) - cv) / fv;
                    camera.distortion->undistort(point);
                    cv::KeyPoint pts = dbInfo.kps[kpt_id];
                    keypoints_db.push_back(pts);
                    pts.pt.x = point(0);
                    pts.pt.y = point(1);
                    keypoints_db_norm.push_back(pts);
                }
                searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, dbInfo.brf_desc,
                                 brief_descriptors_query, keypoints_db, keypoints_db_norm);
                reduceVector(matched_2d_cur, status);
                reduceVector(matched_2d_old, status);
                reduceVector(matched_2d_cur_norm, status);
                reduceVector(matched_2d_old_norm, status);
#if DEBUG_IMG
                    cv::Mat imageQr = cv::imread(image_path_vec[ret[j].Id],
                                                 CV_LOAD_IMAGE_UNCHANGED);
                    vector<cv::Scalar> matched_colors;
                    cv::RNG rng(time(0));
                    for (int ii = 0; ii < static_cast<int>(matched_2d_cur_norm.size()); ++ii) {
                        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
                        matched_colors.push_back(color);
                    }
                    concatImageAndDraw(image_list[i], matched_2d_cur, imageQr, matched_2d_old, matched_colors,
                                       "matched_image", true);

#endif
                FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
                reduceVector(matched_2d_cur, status);
                reduceVector(matched_2d_old, status);
                reduceVector(matched_2d_cur_norm, status);
                reduceVector(matched_2d_old_norm, status);
//                concatImageAndDraw(image_list[i], matched_2d_cur, imageQr, matched_2d_old, matched_colors, "fundamental ransac", true);
                if(matched_2d_cur_norm.size() < MIN_FUNDAMENTAL_THRESHOLD) continue;
                vote_array_fun[i][imageset_id[ret[j].Id].id] += matched_2d_cur_norm.size();
                vote_array_total[imageset_id[ret[j].Id].id] += matched_2d_cur_norm.size();
#if DEBUG_IMG
                concatImageAndDraw(image_list[i], matched_2d_cur, imageQr, matched_2d_old, matched_colors,
                                   "fundransac_matched_image", true);

#endif
            }
        }
    }
    if(DEBUG_INFO_Q) {
        for (int i = 0; i < list_size; i++) {
            cout << "vote_array_fun  ";
            for (int j = 0; j < scene_num; j++) {
                if(vote_array_fun[i][j] == 0) continue;
                cout << j << "-" << vote_array_fun[i][j] << " , ";
            }
            cout << endl;
        }
        cout << "vote_array_total ";

        for (int j = 0; j < scene_num; j++) {
            if(vote_array_total[j] == 0) continue;
            cout << j << "-" <<  vote_array_total[j] << " , ";
        }
        cout << endl;
    }
//    bool notFound = true;
    double score;
    for(int i = 0; i < vote_array_total.size(); i++) {
        score = vote_array_total[i] / (image_list.size() * MIN_FUNDAMENTAL_THRESHOLD * 3);
        if(score > MIN_SCORE) ret_vec.push_back(make_pair(i, RescaleScore(score)));
    }
    sort(ret_vec.begin(), ret_vec.end(),
            [](const pair<int, double> &a,const pair<int, double> &b) -> bool{
        return (a.second > b.second);
    });
    if(DEBUG_INFO_Q) {
        cout << "ret_vec sort: ";
        for (int j = 0; j < ret_vec.size(); j++) {
            cout << ret_vec[j].first << "-" << ret_vec[j].second << " ";
        }
        cout << endl;
    }
    if(ret_vec.size() > 3) ret_vec.resize(3);
    return ret_vec;

}
#if DEBUG_IMG
void ImageDatabase::addImagePath(const string &img_path) {
    image_path_vec.push_back(img_path);
}
#endif

