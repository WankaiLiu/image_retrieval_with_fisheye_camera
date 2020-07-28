//
// Created by wankai on 2020/7/7.
//

#include "ImageDatabase.h"

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

pair<int, double> ImageDatabase::query_list(const std::vector<cv::Mat>& image_list){//根据这一个list中的图片直接在当次判断出当前场景ID
    int window_size=image_list.size();
    std::vector<pair<int,double>> window_id_list;
    std::vector<pair<int,pair<int,double>>> vote_window;
    int trust_id=scene_num+1;
    for (size_t i = 0; i < window_size; i++)//query
    {
        cv::Mat image_blur;
        blurImage4Brief(image_list[i], image_blur);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        computeBRIEFPoint(image_list[i],image_blur,keypoints,brief_descriptors);
        db.query(brief_descriptors, ret, 4, imageset_id.size());

        if (ret.size() >= 1 && ret[0].Score > 0.0001) {
            window_id_list.push_back(make_pair(imageset_id[ret[0].Id],ret[0].Score));
        }
        else {
            window_id_list.push_back(make_pair(scene_num+1,-1));
        }
    }
    sort(window_id_list.begin(), window_id_list.end(),
            [](const pair<int, double> &a, const pair<int, double> &b) {
            return a.second > b.second;
        });//降序排列

    for (size_t i = 0; i < scene_num+2; i++) vote_window.push_back(make_pair(i,make_pair(0,0)));//先把统计窗口中的ID放进去
    for (size_t i = 0; i < window_size; i++) //再把每个ID出现的次数放进去
    {
        if(window_id_list[i].first<scene_num)
        {
            vote_window[window_id_list[i].first].second.first++;
            vote_window[window_id_list[i].first].second.second+=window_id_list[i].second;
        }
    }
    if(window_id_list[0].second/window_id_list[1].second > 2.0)
    vote_window[window_id_list[0].first].second.first+=2;
    // vote_window[last_id].second.first++;
    sort(vote_window.begin(), vote_window.end(),
            [](const pair<int, pair<int,double>> &a, const pair<int, pair<int,double>> &b) {
            return a.second.first > b.second.first;
        });//降序排列

    // if(DEBUG_INFO)
    if(0)
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
        (vote_window[0].second.second/vote_window[0].second.second<2.0))
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
}

void ImageDatabase::extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors) {
    cv::Mat image_blur;
    blurImage4Brief(src, image_blur);
    vector<cv::KeyPoint> keypoints;
    computeBRIEFPoint(src,image_blur,keypoints,brief_descriptors);
}