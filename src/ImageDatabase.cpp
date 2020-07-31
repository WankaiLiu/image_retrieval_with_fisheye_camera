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
    db_info db_info;
    db_info.id = set_id;
    //db_info.index = set_id_index;
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
        return imageset_id[ret[0].Id].id;
    }
    else {
        return -1;
    }
}

void convert_bitset_to_Mat(vector<BRIEF::bitset> i_brf_desc,cv::Mat& o_brf_desc_mat)
{
    o_brf_desc_mat=cv::Mat::zeros(i_brf_desc.size(),32,CV_8UC1);
    int row=0;
    for(vector<BRIEF::bitset> :: iterator iter = i_brf_desc.begin(); iter!=i_brf_desc.end();iter++) {
        BRIEF::bitset bits=*iter;
        for (int i = 0; i < 32; i++){
            char ch=' ';
            int n_offset=i*8;
            for (int j = 0; j < 8; j++) {
                if (bits.test(n_offset + j))	// 第i + j位为1
                    ch |= (1 << j);
                else
                    ch &= ~(1 << j);
            }
            o_brf_desc_mat.at<uchar>(row, i)=(uchar)ch;
        }
        row++;
    }
}

pair<int, double> ImageDatabase::query_list(const std::vector<cv::Mat>& image_list){
    int list_size=image_list.size();
    std::vector<pair<int,double>> window_id_list;
    std::vector<pair<int,pair<int,double>>> vote_window;
    int vote_array[list_size][scene_num];
    int count_result[4][scene_num];

    for (size_t i = 0; i < scene_num; i++) vote_window.push_back(make_pair(i,make_pair(0,0.0)));

    for (int i = 0; i < list_size; i++) {
        for (int j = 0; j < scene_num; j++) {
            vote_array[i][j] = 0;
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < scene_num; j++) {
            count_result[i][j] = 0;
        }
    }
    vector<vector<cv::KeyPoint>> kps_list;
    vector<vector<BRIEF::bitset>> brf_list;
    for (size_t i = 0; i < list_size; i++) {//query
        cv::Mat image_blur;
        blurImage4Brief(image_list[i], image_blur);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        computeBRIEFPoint(image_list[i], image_blur, keypoints, brief_descriptors);
        db.query(brief_descriptors, ret, 4, imageset_id.size());
        kps_list.push_back(keypoints);
        brf_list.push_back(brief_descriptors);
        vector<pair<pair<int,int>,double>> ret4;
        if (ret.size() >= 1 && ret[0].Score > 0.01) {
            for (int j = 0; j < ret.size(); j++) {
                cv::BFMatcher matcher(cv::NORM_HAMMING); 
                std::vector<cv::DMatch> mathces; 
                cv::Mat brf_qr_mat, brf_curt_mat;
                convert_bitset_to_Mat(imageset_id[ret[j].Id].brf_desc,brf_qr_mat);
                convert_bitset_to_Mat(brief_descriptors,brf_curt_mat);
                matcher.match(brf_qr_mat, brf_curt_mat, mathces);
                sort(mathces.begin(), mathces.end(),
                        [](const cv::DMatch &a, const cv::DMatch &b) {
                        return a.distance < b.distance;
                    });
                if(mathces[0].distance < 30)
                {
                    vote_array[i][imageset_id[ret[j].Id].id]++;
                    vote_window[imageset_id[ret[j].Id].id].second.second+=ret[j].Score;
                }
            }
        }
    }

    if(DEBUG_INFO_Q) {
        for (int i = 0; i < list_size; i++) {
            for (int j = 0; j < scene_num; j++) {
                cout << vote_array[i][j] << " ";
            }
            cout << endl;
        }
    }
    for (int i = 0; i < list_size; i++) {
        for (int j = 0; j < scene_num; j++) {
            if (vote_array[i][j] == 4) count_result[0][j]++;
            else if (vote_array[i][j] == 3) count_result[1][j]++;
            else if (vote_array[i][j] == 2) count_result[2][j]++;
            else if (vote_array[i][j] == 1) count_result[3][j]++;
        }
    }

    std::vector<pair<int,double>> weight_statistics;
    for (size_t i = 0; i < scene_num; i++) weight_statistics.push_back(make_pair(i,0));
    const int w[4]={8,6,3,1};

    if(DEBUG_INFO_Q) cout << "------------" << endl;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < scene_num; j++) {
            if(DEBUG_INFO_Q) cout << count_result[i][j] << " ";
            vote_window[j].second.first+=count_result[i][j]*w[i];
            if(count_result[i][j]>0)vote_window[j].second.second*=((double)(5-i))/4;
        }
        if(DEBUG_INFO_Q) cout << endl;
    }

    sort(vote_window.begin(), vote_window.end(),
            [](const pair<int, pair<int,double>> &a, const pair<int, pair<int,double>> &b) {
            if(a.second.first == b.second.first)
                 return a.second.second > b.second.second;
            else return a.second.first > b.second.first;
        });//降序排列
    if(DEBUG_INFO_Q) {
        cout << "------------" << endl;
        for (size_t i = 0; i < scene_num; i++)
        {
            printf("%6d ",  vote_window[i].first);
        }
        cout << endl;
        for (size_t i = 0; i < scene_num; i++)
        {
            printf("%6d ",  vote_window[i].second.first);
        }
        cout << endl;
        for (size_t i = 0; i < scene_num; i++)
        {
            printf("%.4f ",  vote_window[i].second.second);
        }
        cout << endl;
    }

    int x=0;
    if(((double)vote_window[0].second.first/(double)vote_window[1].second.first<1.3)&&
        (vote_window[0].second.second/vote_window[0].second.second<1.3)&&last_id == vote_window[1].first)
        x = 1;
    last_id = vote_window[x].first;

    if(last_id1 == last_id) id_cnt++;
    else id_cnt=0;
    last_id1 = last_id;

    if(vote_window[0].second.first==0)
        return make_pair(-1,0);
    else
        return make_pair(vote_window[x].first,1);

    #if 0
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
    #endif

}

void ImageDatabase::extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors) {
    cv::Mat image_blur;
    blurImage4Brief(src, image_blur);
    vector<cv::KeyPoint> keypoints;
    computeBRIEFPoint(src,image_blur,keypoints,brief_descriptors);
}