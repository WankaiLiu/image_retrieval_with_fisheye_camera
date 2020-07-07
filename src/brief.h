//
// Created by wankai on 2020/7/3.
//
#include "DBoW/DBoW2.h"
#include "DVision/DVision.h"
#include "DBoW/TemplatedDatabase.h"
#include "DBoW/TemplatedVocabulary.h"
using namespace DVision;

#ifndef IMAGE_RETRIEVAL_BRIEF_H
#define IMAGE_RETRIEVAL_BRIEF_H
class BriefExtractor
{
public:
    virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;

    BriefExtractor();
    bool init(const std::string &pattern_file);

    DVision::BRIEF m_brief;
    bool initialed;
#ifdef WITH_FAST
    static PreProcessor m_fastEngine;
#endif
};
#endif //IMAGE_RETRIEVAL_BRIEF_H
