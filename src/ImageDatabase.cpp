//
// Created by wankai on 2020/7/7.
//

#include "ImageDatabase.h"
#define DEBUG_INFO_Q 1
#if 0
#include <opencv2/xfeatures2d/nonfree.hpp>
template <typename T>
void reduceVector(std::vector<T> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
    {
        if (status[i]==1)
            v[j++] = v[i];
    }
    v.resize(j);
}

int minHessian = 400;
int reserve_num = 50;
double chkBySurf(const cv::Mat& img_1,const cv::Mat& img_2)//SURF算法
{
    cv::Ptr<cv::Feature2D> f2d_surf= cv::xfeatures2d::SURF::create(minHessian);
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    f2d_surf->detectAndCompute( img_1, cv::Mat(), keypoints_1, descriptors_1 );
    f2d_surf->detectAndCompute( img_2, cv::Mat(), keypoints_2, descriptors_2 );
    cv::BFMatcher matcher(cv::NORM_L2);
    // cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create(minHessian);
    // vector<KeyPoint> keypoints_1, keypoints_2;
    // f2d->detect(img_1, keypoints_1);
    // f2d->detect(img_2, keypoints_2);
    // Mat descriptors_1, descriptors_2;
    // f2d->compute(img_1, keypoints_1, descriptors_1);
    // f2d->compute(img_2, keypoints_2, descriptors_2);
    // BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);

    nth_element(matches.begin(), matches.begin()+reserve_num, matches.end());   //提取出前最佳匹配结果
    matches.erase(matches.begin()+reserve_num, matches.end());    //剔除掉其余的匹配结果
    std::vector<cv::KeyPoint> kpm_1, kpm_2;
    for( size_t m = 0; m < matches.size(); m++ )
    {
        int i1 = matches[m].queryIdx;
        int i2 = matches[m].trainIdx;
        kpm_1.push_back(keypoints_1[i1]);
        kpm_2.push_back(keypoints_2[i2]);
    }
    std::vector<cv::Point2f> kp1, kp2;
    cv::KeyPoint::convert(kpm_1, kp1);
    cv::KeyPoint::convert(kpm_2, kp2);
    vector<unsigned char> status;
    cv::findFundamentalMat(kp1, kp2, cv::FM_RANSAC, 0.5, 0.99, status);
    reduceVector(kp1, status);
    reduceVector(kp2, status);
    int measure_num = kp1.size();
    double sum_dis=0, avg_dis=0, variance=0;
    std::vector<double> dist;
    for (int i = 0; i < measure_num; i++)
    {
        cv::Point2d diff = cv::Point2d(kp2[i]-kp1[i]);
        double distance = sqrt(diff.x*diff.x+diff.y*diff.y);
        dist.push_back(distance);
        sum_dis+=distance;
    }
    avg_dis = sum_dis / measure_num;
    for (int i = 0; i < measure_num; i++)
    {
        double d = (dist[i]-avg_dis);
        variance += d*d;
    }
    variance /= measure_num;
    return variance;
}
#endif

ImageDatabase::ImageDatabase(string voc_path, std::string _pattern_file){
    BriefVocabulary* voc = new BriefVocabulary(voc_path);
    this->db.setVocabulary(*voc, true, 0);
    pattern_file = _pattern_file;
    delete voc;
    counter = 0;
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
    int MAX_KEYFRAME_KEYPOINTS = 1000;
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
    imageset_id.push_back(set_id);
}

bool ImageDatabase::erase(int id) {
    if(id > imageset_id.size()) {
        return false;
    }
    DBoW2::EntryId entryId = id;
    db.delete_entry(entryId);
    imageset_id.erase(imageset_id.begin() + id);
    return true;
};

int ImageDatabase::query(cv::Mat image){
    cv::Mat image_blur;
    blurImage4Brief(image, image_blur);
    vector<cv::KeyPoint> keypoints;
    vector<BRIEF::bitset> brief_descriptors;
    computeBRIEFPoint(image,image_blur,keypoints,brief_descriptors);
    db.query(brief_descriptors, ret, 4, imageset_id.size());

    if (ret.size() >= 1 && ret[0].Score > 0.0001) {
        cout << "ret[0].Score:" << ret[0].Score << endl;
        return imageset_id[ret[0].Id];
    }
    else {
        return -1;
    }
}

pair<int, double> ImageDatabase::query_list(const std::vector<cv::Mat>& image_list) {//根据这一个list中的图片直接在当次判断出当前场景ID
    int window_size = image_list.size();
    std::vector<pair<int, double>> window_id_list;
    std::vector<pair<int, pair<int, double>>> vote_window;
    int vote_array[window_size][scene_num];
    int count_result[4][scene_num];
//    vector<pair<int,int>> count_result;
//    double vote_score[window_size][scene_num];
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < scene_num; j++) {
            vote_array[i][j] = 0;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < scene_num; j++) {
            count_result[i][j] = 0;
        }
    }
    for (size_t i = 0; i < window_size; i++) {//query
        cv::Mat image_blur;
        blurImage4Brief(image_list[i], image_blur);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        computeBRIEFPoint(image_list[i], image_blur, keypoints, brief_descriptors);
        db.query(brief_descriptors, ret, 4, imageset_id.size());
        if (ret.size() >= 1 && ret[0].Score > 0.0001) {
            for (int j = 0; j < ret.size(); j++) {
                vote_array[i][imageset_id[ret[j].Id]]++;
            }
        }
    }
    if(DEBUG_INFO_Q) {
        for (int i = 0; i < window_size; i++) {
            for (int j = 0; j < scene_num; j++) {
                cout << vote_array[i][j] << " ";
            }
            cout << endl;
        }
    }
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < scene_num; j++) {
            if (vote_array[i][j] == 4) count_result[0][j]++;
            else if (vote_array[i][j] == 3) count_result[1][j]++;
            else if (vote_array[i][j] == 2) count_result[2][j]++;
            else if (vote_array[i][j] == 1) count_result[3][j]++;
        }
    }
if(DEBUG_INFO_Q) {
    cout << "------------" << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < scene_num; j++) {
            cout << count_result[i][j] << " ";
        }
        cout << endl;
    }
}
    int max_id = -1;
    int max_cnt = 0;
    for(int i = 0; i < scene_num; i++) {
        if(count_result[0][i] != 0) {
            max_id = max_cnt < count_result[0][i]? i : max_cnt;
        }
    }
    if(max_id != -1) return pair<int, double> (max_id, 0);
    for(int i = 0; i < scene_num; i++) {
        if(count_result[1][i] != 0) {
            max_id = max_cnt < count_result[1][i]? i : max_cnt;
        }
    }
    if(max_id != -1) return pair<int, double> (max_id, 0);
    for(int i = 0; i < scene_num; i++) {
        if(count_result[2][i] != 0) {
            max_id = max_cnt < count_result[2][i]? i : max_cnt;
        }
    }
    if(max_id != -1) return pair<int, double> (max_id, 0);
    for(int i = 0; i < scene_num; i++) {
        if(count_result[3][i] != 0) {
            max_id = max_cnt < count_result[3][i]? i : max_cnt;
        }
    }
    if(max_id != -1) return pair<int, double> (max_id, 0);

#if 0
    int trust_id=scene_num+1;
    for (size_t i = 0; i < window_size; i++)//query
    {
        cv::Mat image_blur;
        blurImage4Brief(image_list[i], image_blur);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        computeBRIEFPoint(image_list[i],image_blur,keypoints,brief_descriptors);
        db.query(brief_descriptors, ret, 4, imageset_id.size());
        #if 1
        if (ret.size() >= 1 && ret[0].Score > 0.0001) {
            // if(1) printf("%d [0]:%d-%.4f,[1]:%d-%.4f,[2]:%d-%.4f,[3]:%d-%.4f;", i,imageset_id[ret[0].Id], ret[0].Score,\
            imageset_id[ret[1].Id], ret[1].Score, imageset_id[ret[2].Id],ret[2].Score, imageset_id[ret[3].Id],ret[3].Score);
            window_id_list.push_back(make_pair(imageset_id[ret[0].Id],ret[0].Score));
        }
        else {
            window_id_list.push_back(make_pair(scene_num+1,-1));
        }
        // if(1) printf("\n");
        #else
        // for (size_t i = 0; i < 3; i++)
        // {
        //     window_id_list.push_back(make_pair(imageset_id[ret[i].Id],ret[i].Score));
        // }

        window_id_list.push_back(make_pair(imageset_id[ret[0].Id],ret[0].Score*1.2));
        window_id_list.push_back(make_pair(imageset_id[ret[1].Id],ret[1].Score*1));
        window_id_list.push_back(make_pair(imageset_id[ret[2].Id],ret[2].Score*0.8));
        #endif
    }
    sort(window_id_list.begin(), window_id_list.end(),
            [](const pair<int, double> &a, const pair<int, double> &b) {
            return a.second > b.second;
        });//降序排列

    for (size_t i = 0; i < scene_num+2; i++) vote_window.push_back(make_pair(i,make_pair(0,0)));//先把统计窗口中的ID放进去
    for (size_t i = 0; i < window_id_list.size(); i++) //再把每个ID出现的次数放进去
    {
        if(window_id_list[i].first<scene_num)
        {
            vote_window[window_id_list[i].first].second.first++;
            vote_window[window_id_list[i].first].second.second+=window_id_list[i].second;
        }
    }
    if(window_id_list[0].second/window_id_list[1].second > 1.5)
    vote_window[window_id_list[0].first].second.first+=2;
    #if 1
    vote_window[last_id].second.first++;
    sort(vote_window.begin(), vote_window.end(),
            [](const pair<int, pair<int,double>> &a, const pair<int, pair<int,double>> &b) {
            if(a.second.first == b.second.first)
                 return a.second.second > b.second.second;
            else return a.second.first > b.second.first;
        });//降序排列
    #else
    sort(vote_window.begin(), vote_window.end(),
            [](const pair<int, pair<int,double>> &a, const pair<int, pair<int,double>> &b) {
            if(a.second.first == b.second.first)
                 return a.second.second > b.second.second;
            else return a.second.first > b.second.first;
        });//降序排列
    #endif

    // if(DEBUG_INFO)
    if(1)
    {
        printf("-----------------------------------------------------------------------(%d %.4f),(%d %.4f),(%d %.4f)  last:%d,%d\n",
        window_id_list[0].first,window_id_list[0].second,
        window_id_list[2].first,window_id_list[2].second,
        window_id_list[3].first,window_id_list[3].second,
        last_id,id_cnt);
        printf("vote ID:  ");
        for (size_t i = 0; i < vote_window.size(); i++) printf("   %4d",vote_window[i].first);
        printf("\nvote cnt: ");
        for (size_t i = 0; i < vote_window.size(); i++) printf("   %4d",vote_window[i].second.first);
        printf("\nvote scr: ");
        for (size_t i = 0; i < vote_window.size(); i++) printf("   %.2f",vote_window[i].second.second);
        printf("\n");
    }

    double confidence1 = (double)vote_window[0].second.first/(double)window_size;//置信度最高
    double confidence2 = (double)vote_window[1].second.first/(double)window_size;

    int x=0;
    if(((double)vote_window[0].second.first/(double)vote_window[1].second.first<2.0)&&
        (vote_window[0].second.second/vote_window[0].second.second<2.0))//&&confidence1<0.5)
    for (size_t i = 0; i < 5; i++)
    {
        if(id_cnt < 1)
        {
            if(vote_window[i].first == last_id)
            {
                x = i;
                break;
            }
        }
        else
        {
            if(vote_window[i].first == last_id1)
            {
                x = i;
                break;
            }
        }
    }
    // printf("last:%d  x:%d  cur:%d\n",last_id,x,vote_window[x].first);
    last_id = vote_window[x].first;

    if(last_id1 == last_id) id_cnt++;
    else id_cnt=0;
    last_id1 = last_id;

    double confidence = (double)vote_window[x].second.first/(double)window_size;

    return make_pair(vote_window[x].first, confidence);
#endif
}

void ImageDatabase::extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors) {
    cv::Mat image_blur;
    blurImage4Brief(src, image_blur);
    vector<cv::KeyPoint> keypoints;
    computeBRIEFPoint(src,image_blur,keypoints,brief_descriptors);
}