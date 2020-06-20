#include "dig_stabilizer.h"

const cv::Size sub_pix_win_size(3,3);
const cv::Size opt_flow_win_size(5,5);
const cv::TermCriteria termcrit(cv::TermCriteria::MAX_ITER, 100, 0.01);

dig_stabilizer::dig_stabilizer(int sample_len, Point2f img_size, float lf_alpha)
{
    sz = img_size;
    alpha = lf_alpha;
    //video_seq_coord
    sample_depth = sample_len;
}

void dig_stabilizer::next_transform(Point2f offset)
{
    if (video_seq_coord.size() < sample_depth) // not full fifo yet
    {
        if (video_seq_coord.size())
        {
            Point2f buf = video_seq_coord.back();
            buf += offset;
            video_seq_coord.push_back(buf);
        }
        else
        {
            Point2f buf(0.0, 0.0);
            video_seq_coord.push_back(buf);
        }
    }
    else
    {
        Point2f ref = video_seq_coord.at(1);
        for (auto &p: video_seq_coord)
            p -= ref;
        std::copy(video_seq_coord.begin()+1, video_seq_coord.end(), video_seq_coord.begin());
        Point2f buf = video_seq_coord.at(sample_depth-2);
        buf += offset;
        video_seq_coord.back() = buf;
    }
}

Point2f dig_stabilizer::get_roi_offset(bool &ok)
{
    ok = true;
    Point2f filtered_position = video_seq_coord.at(0);
    for (int i = 1; i < video_seq_coord.size(); i++)
    {
        filtered_position = filtered_position * alpha + video_seq_coord.at(i) * (1.0 - alpha);
    }
    Point2f stab_shift = filtered_position - video_seq_coord.back();
    return stab_shift;
}

Point2f dig_stabilizer::get_mean_shift(vector<Point2f> &start, vector<Point2f> &stop)
{
    Point2f acc(0.0f, 0.0f);
    for(size_t i = 0; i < start.size(); i++)
        acc += stop[i] - start[i];
    acc = acc / float(start.size());
    return acc*(-1); // Real offset is opposite to optical flow
}

float dig_stabilizer::get_mean_rotation(vector<cv::Point2f> &start, vector<cv::Point2f> &stop)
{
     using std::cout;
     Mat H = findHomography(start, stop);
     Mat R1, R2, T;
     decomposeEssentialMat(H, R1, R2, T);
     float * r1 = reinterpret_cast<float *>(R1.data);
     float * r2 = reinterpret_cast<float *>(R2.data);
     float * t = reinterpret_cast<float *>(T.data);


     cout << "DECOMPOSITION****************************" << std::endl;
     cout << "R1: " << R1.cols << " " << R1.rows << std:: endl;
     for (int i = 0; i < 9; i++)
         cout << r1[i] << " ";
     cout << std::endl;
     cout << "R2: " << R2.cols << " " << R2.rows << std:: endl;
     for (int i = 0; i < 9; i++)
         cout << r2[i] << " ";
     cout << std::endl;
     cout << "t: " << T.cols << " " << T.rows << std:: endl;
     for (int i = 0; i < 3; i++)
         cout << t[i] << " ";
     cout << std::endl;

}
/*
float dig_stabilizer::get_mean_rotation(vector<Point2f> &start, vector<Point2f> &stop)
{
    const int iteration_limit = 1;
    const float common_share = 0.5;
    const int probes_limit = 10;
    const float PI = 3.1415926;

    Mat pts0(3, 2, CV_32F);
    Mat pts1(3, 2, CV_32F);
    float * matp0 = reinterpret_cast<float *>(pts0.data);
    float * matp1 = reinterpret_cast<float *>(pts1.data);
    int p0_id,p1_id,p2_id;

    std::cout << " ANGLES LAB :" << std::endl;

    for (int i = 0; i < iteration_limit; i++)
    {
        p0_id = rand(0, start.size()-1);
        p1_id = rand(0, start.size()-1);
        p2_id = rand(0, start.size()-1);
        // Надо сдвинуть координаты и навести порядок
        matp0[3*0 + 0] = start[p0_id].x - imgsz.x/2;
        matp0[3*0 + 1] = start[p0_id].y - imgsz.y/2;
        matp0[3*1 + 0] = start[p1_id].x - imgsz.x/2;
        matp0[3*1 + 1] = start[p1_id].y - imgsz.y/2;;
        matp0[3*2 + 0] = start[p2_id].x - imgsz.x/2;
        matp0[3*2 + 1] = start[p2_id].y - imgsz.y/2;;
        matp1[3*0 + 0] = stop[p0_id].x - imgsz.x/2;
        matp1[3*0 + 1] = stop[p0_id].y - imgsz.y/2;;
        matp1[3*1 + 0] = stop[p1_id].x - imgsz.x/2;
        matp1[3*1 + 1] = stop[p1_id].y - imgsz.y/2;;
        matp1[3*2 + 0] = stop[p2_id].x - imgsz.x/2;
        matp1[3*2 + 1] = stop[p2_id].y - imgsz.y/2;;
        Mat M = getAffineTransform(pts0, pts1);
        float * matm = reinterpret_cast<float *>(M.data);

        auto alpha_0 = asin(-1*matm[3*0+1]) * 180 / PI;
        auto alpha_1 = acos(matm[3*0+0]) * 180 / PI;
        auto alpha_2 = asin(matm[3*1+0]) * 180 / PI;
        auto alpha_3 = acos(matm[3*1+1]) * 180 / PI;

        float angle = 0;
        float okcnt = 0;
        if (!isnan(alpha_0))
        {
            angle+=alpha_0; okcnt++;
            std::cout << angle << " ";
        }
        if (!isnan(alpha_1))
        {
            angle+=alpha_1; okcnt++;
            std::cout << angle << " ";
        }
        if (!isnan(alpha_2))
        {
            angle+=alpha_2; okcnt++;
            std::cout << angle << " ";
        }
        if (!isnan(alpha_3))
        {
            angle+=alpha_3; okcnt++;
            std::cout << angle << " ";
        }
        if (okcnt)
            angle /= okcnt;

        std::cout << angle << " ";
    }
    std::cout << std::endl;
}
*/
void dig_stabilizer::set_frames(const Mat &prev_gray, const Mat &cur_gray)
{
    prev_frame = prev_gray.clone();
    cur_frame = cur_gray.clone();
}

void dig_stabilizer::set_src_image(const Mat &src_img)
{
    source_image = src_img.clone();
    imgsz = Point2f(src_img.cols, src_img.rows);
}

void dig_stabilizer::process()
{
    // find good features on the previous frame
    cv::goodFeaturesToTrack(prev_frame, prev_feat_points, 1000, 0.01, 10.0, cv::noArray(), 3, false, 0.04);
    // refine them
    if(!prev_feat_points.empty())
    {
       cornerSubPix(prev_frame, prev_feat_points, sub_pix_win_size, cv::Size(-1,-1), termcrit);
    }
    // if there are some feature points on the previous frame
    if(!prev_feat_points.empty())
    {
        // start to compute optflow in 2 directions
        std::vector<float> err;
        calcOpticalFlowPyrLK(prev_frame, cur_frame, prev_feat_points, cur_optfl_points, cur_optfl_status, err, opt_flow_win_size, 3, termcrit, 0, 0.001);
        calcOpticalFlowPyrLK(cur_frame, prev_frame, cur_optfl_points, backtrace_optfl_points, backtrace_optfl_status, err, opt_flow_win_size, 3, termcrit, 0, 0.001);
        // assemble output point pairs
        prev_points.clear();
        prev_points.resize(0);
        cur_points.clear();
        cur_points.resize(0);
        for(size_t i = 0; i < prev_feat_points.size(); i++)
        {
            if (cur_optfl_status[i] && backtrace_optfl_status[i])
            {
                if (cv::norm(prev_feat_points[i] - backtrace_optfl_points[i]) <= 20)
                {
                    prev_points.push_back(prev_feat_points[i]);
                    cur_points.push_back(cur_optfl_points[i]);
                }
                else
                    continue;
            }
            else
                continue;
        }
    }
    Point2f mean_shift;
    float mean_rotation = 0.0;
    if (prev_points.size() > 3) // can trust to results
    {
        mean_shift = get_mean_shift(prev_points, cur_points);
        mean_rotation = get_mean_rotation(prev_points, cur_points);
    }
    else
    {
        mean_shift = Point2f(0,0);
        mean_rotation = 0.0;
    }

    std::cout << "Stable shift:\tX: " << mean_shift.x << "\t Y: " << mean_shift.y << std::endl;

    bool ok;
    next_transform(mean_shift);
    Point2f roff = get_roi_offset(ok);

    Point2i img_sz_int(int(imgsz.x), int(imgsz.y));
    Point2i stab_shift_int(int(roff.x), int(roff.y));

    // find correct roi on the current frame
    Point2i roi_top_left, roi_bottom_right;
    roi_top_left = Point2i(0, 0);
    roi_bottom_right = Point2i(int(imgsz.x)-1, int(imgsz.y)-1);
    roi_top_left += stab_shift_int;
    roi_bottom_right += stab_shift_int;
    validate_img_point(roi_top_left, img_sz_int);
    validate_img_point(roi_bottom_right, img_sz_int);
    cur_frame_roi = Rect(roi_top_left.x, roi_top_left.y,
                      roi_bottom_right.x - roi_top_left.x + 1,
                      roi_bottom_right.y - roi_top_left.y + 1);

    // find output roi (position on the frame) for stab frame
    Point2i origin = stab_shift_int * (-1);
    validate_img_point(origin, img_sz_int);
    stab_frame_roi = Rect(origin.x, origin.y, cur_frame_roi.width, cur_frame_roi.height);

    // visualization
    for(size_t i = 0; i < prev_points.size(); i++)
    {
       Point2f p = cur_points[i] - prev_points[i];
       circle(source_image, cur_points[i], 3, cv::Scalar(0,255,0), -1, 8);
       cv::line(source_image, prev_points[i], cur_points[i], cv::Scalar(0,0,255), 1);
    }
    show_frame("src stream", source_image);
}

Rect dig_stabilizer::get_cur_roi()
{
    return cur_frame_roi;
}

Rect dig_stabilizer::get_stab_roi()
{
    return stab_frame_roi;
}

void dig_stabilizer::show_frame(string label, Mat &f)
{
    if (f.empty())
        return;
    imshow(label, f);
}

void dig_stabilizer::validate_img_point(Point2i &p, Point2i &limits)
{
    if (p.x < 0)
        p.x = 0;
    if (p.y < 0)
        p.y = 0;
    if (p.x > limits.x - 1)
        p.x = limits.x - 1;
    if (p.y > limits.y - 1)
        p.y = limits.y - 1;
}

int dig_stabilizer::rand(int first, int last)
{
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(first,last); // guaranteed unbiased
    auto random_integer = uni(rng);
    return random_integer;
}
