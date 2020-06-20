#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "dig_stabilizer.h"

using namespace cv;
using std::string;
using std::cout;
using time_point = std::chrono::high_resolution_clock::time_point;
using cv::Point2f;
using cv::Point2i;
using cv::Rect;

// util function for convinient msg io
auto prnt = [] (string msg) {
    std::cout << msg << std::endl;
};

// util for stream show
void show_frame(string label, Mat &f)
{
    if (f.empty())
        return;
    imshow(label, f);
}

// util function for capturer check
auto check_cap = [] (VideoCapture &c, string label) {
    bool ok = true;
    if (!c.isOpened()) {
        prnt("Error while opening "+label+" file");
        ok = false;
    }
    return ok;
};

time_point timestamp()
{
    time_point tstamp = std::chrono::high_resolution_clock::now();
    return tstamp;
}

long int interval_us(const time_point &start, const time_point &stop)
{
    long int interval =  std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return interval;
}

void validate_img_point(Point2i &p, Point2i &limits)
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


const char vpath[] = "/home/igor/nf3.avi";

int main(int argc, char *argv[])
{
    VideoCapture webcam(0);
    if (!check_cap(webcam, "web-camera"))
           return -1;

    Mat src_frame, cur_gray, prev_gray;
    Mat stab_frame;
    int id = 0;
    time_point it_start, it_end;
    Point2f imgsz(640, 480);


    dig_stabilizer stab(50, imgsz, 0.9);

    Size dest_sz(imgsz.x, imgsz.y);
    stab_frame = Mat(imgsz.y, imgsz.x, CV_8UC3);
    Mat compare_frame = Mat(imgsz.y, imgsz.x, CV_8UC3);
    Rect unstable_roi(0,0,imgsz.x,imgsz.y);
    int cent_square = 200;
    Rect stable_roi(imgsz.x/2-cent_square/2,imgsz.y/2-cent_square/2,cent_square,cent_square);


    while(true)
    {
        it_start = timestamp();
        webcam >> src_frame;
        if (src_frame.empty())
           break;
        resize(src_frame,src_frame, dest_sz);
        cv::cvtColor(src_frame, cur_gray, cv::COLOR_BGR2GRAY);

        if (!cur_gray.empty() && !prev_gray.empty())
        {
            stab.set_src_image(src_frame);
            stab.set_frames(prev_gray, cur_gray);
            stab.process();
            auto cur_frame_stab_roi = stab.get_cur_roi();
            auto out_frame_stab_roi = stab.get_stab_roi();

            // clear frame
            stab_frame.setTo(0);

            auto roi = src_frame(cur_frame_stab_roi).clone();
            auto dest = stab_frame(out_frame_stab_roi);

            roi.copyTo(dest);

            auto unstable = src_frame(unstable_roi);
            auto stable = stab_frame(stable_roi);
            auto dest_unst = compare_frame(unstable_roi);
            auto dest_st = compare_frame(stable_roi);
            unstable.copyTo(dest_unst);
            stable.copyTo(dest_st);
        }
        // whatever: copy cur frame to prev
        prev_gray = cur_gray.clone();

        show_frame("stab stream", stab_frame);
        show_frame("comparsion stream", compare_frame);


        it_end = timestamp();
        auto us_full = interval_us(it_start, it_end);
        prnt("=== Iteration summary ===");
        cout << "Frame ID : " << id << "\tFull time : " << us_full << std::endl;
        id++;

        char c = (char)waitKey(1);
        if(c=='c')
           break;
    }

    return 0;
}
