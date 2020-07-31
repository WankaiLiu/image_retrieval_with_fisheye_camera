//
// Created by wankai on 2020/7/7.
//

#include "ImageDatabase.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

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
    int MAX_KEYFRAME_KEYPOINTS = 600;
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

#ifdef DEBUG
void ImageDatabase::addImage(const cv::Mat &image, int set_id, int set_id_index) {
    cv::Mat image_blur;
    blurImage4Brief(image, image_blur);
    vector<cv::KeyPoint> keypoints;
    vector<BRIEF::bitset> brief_descriptors;
    computeBRIEFPoint(image,image_blur,keypoints,brief_descriptors);
    db.add(brief_descriptors);
    // imageset_id.push_back(make_pair(set_id,set_id_index));
    db_info db_info;
    db_info.id = set_id;
    db_info.index = set_id_index;
    db_info.kps = keypoints;
    db_info.brf_desc = brief_descriptors;
    imageset_id.push_back(db_info);
}
#else
void ImageDatabase::addImage(const cv::Mat &image, int set_id) {
    cv::Mat image_blur;
    blurImage4Brief(image, image_blur);
    vector<cv::KeyPoint> keypoints;
    vector<BRIEF::bitset> brief_descriptors;
    computeBRIEFPoint(image,image_blur,keypoints,brief_descriptors);
    db.add(brief_descriptors);
    imageset_id.push_back(set_id);
}
#endif

bool ImageDatabase::erase(int id) {
    if(id > imageset_id.size()) {
        return false;
    }
    DBoW2::EntryId entryId = id;
    db.delete_entry(entryId);
    imageset_id.erase(imageset_id.begin() + id);
    return true;
};


#ifdef DEBUG
void convert_bitset_to_Mat(vector<BRIEF::bitset> temp_brief_descriptors,cv::Mat& out_brief_descriptors_mat)
{
    //2.convert bitset to Mat
    out_brief_descriptors_mat=cv::Mat::zeros(temp_brief_descriptors.size(),32,CV_8UC1);
    int row=0;
    for(vector<BRIEF::bitset> :: iterator iter = temp_brief_descriptors.begin(); iter!=temp_brief_descriptors.end();iter++)
    {
        BRIEF::bitset bits=*iter;

        for (int i = 0; i < 32; i++)
        {
            char ch=' ';
            int n_offset=i*8;
            for (int j = 0; j < 8; j++)
            {
                if (bits.test(n_offset + j))	// 第i + j位为1
                    ch |= (1 << j);
                else
                    ch &= ~(1 << j);
            }
            out_brief_descriptors_mat.at<uchar>(row, i)=(uchar)ch;
        }
        row++;
    }

}

pair<int,int> ImageDatabase::query(cv::Mat image){
    cv::Mat image_blur;
    blurImage4Brief(image, image_blur);
    vector<cv::KeyPoint> keypoints;
    vector<BRIEF::bitset> brief_descriptors;
    computeBRIEFPoint(image,image_blur,keypoints,brief_descriptors);
    db.query(brief_descriptors, ret, 4, imageset_id.size());

    if (ret.size() >= 1 && ret[0].Score > 0.005) {
        cout << "ret[0].Score:" << ret[0].Score << endl;
        printf("%d, %d\n",imageset_id[ret[0].Id].id,ret[0].Id);
        return make_pair(imageset_id[ret[0].Id].id,imageset_id[ret[0].Id].index);
    }
    else {
        return make_pair(-1,-1);
    }
}

pair<int, double> ImageDatabase::query_list(int set_id, const std::vector<cv::Mat>& image_list){//根据这一个list中的图片直接在当次判断出当前场景ID
    int list_size=image_list.size();
    std::vector<pair<int,double>> window_id_list;
    std::vector<pair<int,pair<int,double>>> vote_window;
    std::vector<pair<int,int>> q_id0;
    std::vector<vector<pair<pair<int,int>,double>>> q_ret4;
    int trust_id=scene_num+1;
    int high_rate_cnt=0;

     #if 1

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
        if (ret.size() >= 1 && ret[0].Score > 0.001) {
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
                    ret4.push_back(make_pair(make_pair(imageset_id[ret[j].Id].id,imageset_id[ret[j].Id].index),ret[j].Score));
                }
            }
            q_id0.push_back(make_pair(imageset_id[ret[0].Id].id,imageset_id[ret[0].Id].index));
            q_ret4.push_back(ret4);
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
    if(DEBUG_INFO_Q) {
        cout << "------------" << endl;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < scene_num; j++) {
                cout << count_result[i][j] << " ";

                vote_window[j].second.first+=count_result[i][j]*w[i];

                if(count_result[i][j]>0)vote_window[j].second.second*=((double)(5-i))/4;
            }
            cout << endl;
        }
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
    
    #if 1
    printf("----------------%d-%d\n",set_id,vote_window[0].first);
    // if(vote_window[0].first!=set_id)
    #if 0
    {
        for(int i=0; i< list_size; i++)
        {
            for (size_t j = 0; j < 4; j++)
            {
                int q_id = q_ret4[i][j].first.first;
                int q_index = q_ret4[i][j].first.second;
                vector<cv::KeyPoint> kps_qr;
                vector<BRIEF::bitset> brf_qr;
                for (size_t i = 0; i < imageset_id.size(); i++)
                {
                    if((q_id == imageset_id[i].id)&&(q_index == imageset_id[i].index))
                    {
                        kps_qr = imageset_id[i].kps;
                        brf_qr = imageset_id[i].brf_desc;
                    }
                }

                cv::BFMatcher matcher(cv::NORM_HAMMING); 
                std::vector<cv::DMatch> mathces; 
                cv::Mat brf_qr_mat, brf_list_i_mat;
                convert_bitset_to_Mat(brf_qr,brf_qr_mat);
                convert_bitset_to_Mat(brf_list[i],brf_list_i_mat);
                matcher.match(brf_qr_mat, brf_list_i_mat, mathces);
                sort(mathces.begin(), mathces.end(),
                        [](const cv::DMatch &a, const cv::DMatch &b) {
                        return a.distance < b.distance;
                    });//sheng序排列
                // double sum_dis=0;
                // for (size_t i = 0; i < 30; i++)
                // {
                //     sum_dis += mathces[i].distance;
                //     printf("%.0f  ",mathces[i].distance);
                // }
                // printf("\n");

                printf("=========set_id:%d, current list[%d] - q_ret[%d].id:%d score:%.4f | mth:%f\n",set_id, i , j, q_id, q_ret4[i][j].second,mathces[0].distance);
                cv::Mat imageQr = cv::imread(images_DBList[q_id].second[q_index], CV_LOAD_IMAGE_UNCHANGED);
                cv::Mat im_list;
                cvtColor(image_list[i], im_list, cv::COLOR_GRAY2RGB);
                cv::drawKeypoints(im_list,kps_list[i],im_list);
                cv::Mat img_qr, show_img;
                cvtColor(imageQr, img_qr, cv::COLOR_GRAY2RGB);
                cv::drawKeypoints(img_qr,kps_qr,img_qr);
                cv::hconcat(img_qr, im_list, show_img);
                char buf1[10],buf2[10];
                sprintf(buf1, "ret: %d", q_id);
                cv::putText(show_img, buf1, cv::Point(10,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                sprintf(buf2, "set_id: %d", set_id);
                cv::putText(show_img, buf2, cv::Point(650,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                cv::imshow("matched", show_img);
                cv::waitKey(0);
            }
        }
    }
    #endif
    #endif
    if(vote_window[0].second.first==0)
        return make_pair(-1,0);
    else
        return make_pair(vote_window[x].first,0);

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

    #else
    for (size_t i = 0; i < list_size; i++)//query
    {
        cv::Mat image_blur;
        blurImage4Brief(image_list[i], image_blur);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        computeBRIEFPoint(image_list[i],image_blur,keypoints,brief_descriptors);
        db.query(brief_descriptors, ret, 4, imageset_id.size());
        printf("%d|   %d-%.4f,  %d-%.4f,  %d-%.4f,  %d-%.4f;\n", i,
            imageset_id[ret[0].Id].first, ret[0].Score, imageset_id[ret[1].Id].first, ret[1].Score, 
            imageset_id[ret[2].Id].first, ret[2].Score, imageset_id[ret[3].Id].first, ret[3].Score);
        if (ret.size() >= 1 && ret[0].Score > 0.0001){
            window_id_list.push_back(make_pair(imageset_id[ret[0].Id].first,ret[0].Score));
            q_id0.push_back(imageset_id[ret[0].Id]);
        }
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
                 return a.second.second > b.second.second;
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

    double confidence1 = (double)vote_window[0].second.first/(double)list_size;//置信度最高
    double confidence2 = (double)vote_window[1].second.first/(double)list_size;

    printf("----------------%d-%d\n",set_id,vote_window[0].first);
    if(vote_window[0].first!=set_id)
    {
        for(int i=0; i< list_size; i++)
        {
            printf("<<<<<<<<<<%d-%d\n",set_id,q_id0[i].first);
            cv::Mat imageDB = cv::imread(images_DBList[q_id0[i].first].second[q_id0[i].second], CV_LOAD_IMAGE_UNCHANGED);
            cv::Mat img, show_img;
            // imgSIFT(image, imageDB);
            // imgSURF(image, imageDB);
            cv::hconcat(image_list[i], imageDB, show_img);
            // img.convertTo(show_img, CV_8UC3);
            // cvCvtColor(&img, &show_img, cv::COLOR_GRAY2RGB);
            char buf1[10],buf2[10];
            sprintf(buf1, "test: %d", q_id0[i].first);
            cv::putText(show_img, buf1, cv::Point(10,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            sprintf(buf2, "query: %d", set_id);
            cv::putText(show_img, buf2, cv::Point(650,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            cv::imshow("matched", show_img);
            cv::waitKey(0);
        }
    }
    return make_pair(vote_window[0].first, confidence1);
    #endif
}

void ImageDatabase::saveImagePath(const int id, const vector<string> &imagesList) {
    images_DBList.push_back(make_pair(id,imagesList));
}
#else
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
    int list_size = image_list.size();
    std::vector<pair<int, double>> window_id_list;
    std::vector<pair<int, pair<int, double>>> vote_window;
    int vote_array[list_size][scene_num];
    int count_result[4][scene_num];

    #if 1

    for (size_t i = 0; i < scene_num; i++) vote_window.push_back(make_pair(i,make_pair(0,0.0)));
//    vector<pair<int,int>> count_result;
//    double vote_score[list_size][scene_num];
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
    for (size_t i = 0; i < list_size; i++) {//query
        cv::Mat image_blur;
        blurImage4Brief(image_list[i], image_blur);
        vector<cv::KeyPoint> keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        computeBRIEFPoint(image_list[i], image_blur, keypoints, brief_descriptors);
        db.query(brief_descriptors, ret, 4, imageset_id.size());
        if (ret.size() >= 1 && ret[0].Score > 0.0001) {
            for (int j = 0; j < ret.size(); j++) {
                vote_array[i][imageset_id[ret[j].Id]]++;
                // vote_window[imageset_id[ret[j].Id]].second.first++;
                vote_window[imageset_id[ret[j].Id]].second.second+=ret[j].Score;
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
    if(DEBUG_INFO_Q) {
        cout << "------------" << endl;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < scene_num; j++) {
                cout << count_result[i][j] << " ";
                // weight_statistics[j].second+=count_result[i][j]*(4-i);

                vote_window[j].second.first+=count_result[i][j]*w[i];

                if(count_result[i][j]>0)vote_window[j].second.second*=((double)(5-i))/4;
            }
            cout << endl;
        }

        // cout << "------------" << endl;
        // for (size_t i = 0; i < scene_num; i++)
        // {
        //     printf("%4d ",  weight_statistics[i].first);
        // }
        // cout << endl;
        // for (size_t i = 0; i < scene_num; i++)
        // {
        //     printf("%.2f ",  weight_statistics[i].second);
        // }
        // cout << endl;
        
    }

    // sort(weight_statistics.begin(), weight_statistics.end(),
    //         [](const pair<int, double> &a, const pair<int, double> &b) {
    //         return a.second > b.second;
    //     });//降序排列
        
    // return make_pair(weight_statistics[0].first,weight_statistics[0].second/40);

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
            printf("%4d ",  vote_window[i].first);
        }
        cout << endl;
        for (size_t i = 0; i < scene_num; i++)
        {
            printf("%4d ",  vote_window[i].second.first);
        }
        cout << endl;
        for (size_t i = 0; i < scene_num; i++)
        {
            printf("%.2f ",  vote_window[i].second.second);
        }
        cout << endl;
    }

    printf("lastid:%d\n",  last_id);

    // int x=0;
    // if(((double)vote_window[0].second.first / (double)vote_window[1].second.first < 1.3) &&
    //     (vote_window[0].second.second / vote_window[1].second.second < 1.3))
    // {
    //     if(vote_window[1].first == last_id) x = 1;
    // }
    // last_id = vote_window[x].first;

    // if(last_id1 == last_id) id_cnt++;
    // else id_cnt=0;
    // last_id1 = last_id;

    return make_pair(vote_window[0].first,0);

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

    #else
    int trust_id=scene_num+1;
    for (size_t i = 0; i < list_size; i++)//query
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

    double confidence1 = (double)vote_window[0].second.first/(double)list_size;//置信度最高
    double confidence2 = (double)vote_window[1].second.first/(double)list_size;

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

    double confidence = (double)vote_window[x].second.first/(double)list_size;

    return make_pair(vote_window[x].first, confidence);
    #endif
}

void ImageDatabase::extractFeatureVector(const cv::Mat &src, vector<BRIEF::bitset> &brief_descriptors) {
    cv::Mat image_blur;
    blurImage4Brief(src, image_blur);
    vector<cv::KeyPoint> keypoints;
    computeBRIEFPoint(src,image_blur,keypoints,brief_descriptors);
}
#endif