#include "brief.h"
BriefExtractor::BriefExtractor()
{
    initialed = false;
}

bool BriefExtractor::init(const std::string &pattern_file)
{
    if (initialed)
    {
        return true;
    }
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary

    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw string("Could not open file ") + pattern_file;
    }

    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);

    initialed = true;

#ifdef WITH_FAST
    LOGD("m_fastEngine init: %d, %d\n", COL, ROW);
    m_fastEngine.Init(COL, ROW, 2000);
#endif

    return true;
}

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
#ifdef WITH_FAST
    cv::Mat _descriptors(keys.size(), 32, CV_8U);
    m_fastEngine.Processing(im, keys, _descriptors);
    LOGD("m_fastEngine processing\n");

    descriptors.resize(keys.size());
    std::vector<DVision::BRIEF::bitset>::iterator dit;
    dit = descriptors.begin();
    for(int i = 0; i < keys.size(); ++i, ++dit)
    {
        cv::Mat r = _descriptors.row(i);
        dit->resize(256);
        dit->reset();
        boost::from_array_range((uint64*)r.data, 4, *dit);
    }

#else
    m_brief.compute(im, keys, descriptors, false);
#endif
}
