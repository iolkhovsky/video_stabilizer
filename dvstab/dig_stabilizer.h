#ifndef DIG_STABILIZER_H
#define DIG_STABILIZER_H

#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <random>

using std::string;
using cv::Point2f;
using cv::Point2i;
using cv::Rect;
using std::vector;
using cv::Mat;

class dig_stabilizer
{
private:
    vector<Point2f> video_seq_coord;
    Point2f sz;
    float alpha;
    int sample_depth;
    Mat prev_frame;
    Mat cur_frame;
    Mat source_image;
    std::vector<Point2f> prev_feat_points;
    std::vector<Point2f> cur_optfl_points;
    std::vector<uchar> cur_optfl_status;
    std::vector<Point2f> backtrace_optfl_points;
    std::vector<uchar> backtrace_optfl_status;
    std::vector<Point2f> prev_points;
    std::vector<Point2f> cur_points;
    Rect cur_frame_roi;
    Rect stab_frame_roi;

    void show_frame(string label, Mat &f);
    void validate_img_point(Point2i &p, Point2i &limits);
    int rand(int first, int last);
    Point2f imgsz;

public:
    dig_stabilizer(int sample_len, const Point2f img_size, float lf_alpha);
    void next_transform(Point2f offset);
    Point2f get_roi_offset(bool &ok);
    Point2f get_mean_shift(vector<Point2f> &start, vector<Point2f> &stop);
    float get_mean_rotation(vector<Point2f> &start, vector<Point2f> &stop);
    void set_frames(const Mat &prev_gray, const Mat &cur_gray);
    void set_src_image(const Mat &src_img);
    void process();
    Rect get_cur_roi();
    Rect get_stab_roi();
};

#endif // DIG_STABILIZER_H
