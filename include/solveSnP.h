/*
 * Estimates position of a 360 degrees camera using 3D points and image points correspondences.
 * Uses spherical projection geometry as described below:
 * phi = (iy - _image_center) / _scale_factor
 * theta = (ix - _image_center) / _scale_factor
 * cx = sin(theta)*cos(phi)
 * cy = sin(phi)
 * cz = cos(theta)*cos(phi)
 * 
 * where ix and iy are image coordinates
 * and cx, cy and cz are 3D coordinated of the corresponding point in camera frame
 * 
 * Author: Manash Pratim Das (mpdmanash@iitkgp.ac.in) 
 */

#ifndef SOLVESNP_H
#define SOLVESNP_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <omp.h>

//#define M_PI 3.14159265

class SolveSnP{
public:
    SolveSnP(int image_height);
    ~SolveSnP();
    void Solve(std::vector<cv::DMatch> & in_matches, std::vector<cv::Vec3f> & in_kps3d, std::vector<cv::KeyPoint> & in_kps,
               int snp_pairs, int num_ransac, double threshold, double * final_camera, std::vector<int> & out_inliers);
    void Camera2World(double cx, double cy, double cz, double * w2cT, double & wx, double & wy, double & wz);
    void Camera2Image(double cx, double cy, double cz, int & ix, int & iy);
    void Image2Sphere(int x, int y, double & sx, double & sy, double & sz);
    void World2Sphere(double x, double y, double z, double * w2cT, double & sx, double & sy, double & sz);
    
private:
    void _RunOptimizer(int num_pairs, double s_obs[][3], double world_obs[][3], double * out_cam, bool use_initial=false);
    void _FindInliers(std::vector<cv::DMatch> & in_matches, std::vector<cv::Vec3f> & in_kps3d,
                      std::vector<cv::KeyPoint> & in_kps, double * solved_camera, double threshold,
                      int & out_num_inliers, std::vector<int> & out_in_ids);
    
    int _image_height;
    int _image_center;
    double _scale_factor;
};

#endif
