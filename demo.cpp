#include <iostream>
#include "solveSnP.h"
#include<cstdlib>
#include <time.h>

int main(){
  std::srand (time(NULL));
  SolveSnP solver(480);  
  int points = 700;
  
  // Generate synthetic 3D points with respect to camera frame
  std::vector<cv::Vec3d> cp, wp;
  std::vector<cv::Vec2i> ip;
  for(int i=0; i<points; i++)
  {
    double x = rand() % (20 - (-20)) + (-20);
    double y = rand() % (20 - (-20)) + (-20);
    double z = rand() % (20 - (-20)) + (-20);
    cp.push_back(cv::Vec3d(x, y, z));
    wp.push_back(cv::Vec3d(0, 0, 0));
    ip.push_back(cv::Vec2i(0,0));
  }
  
  // Apply transformation T to move them to world frame
  double w2cT [6] = {0, 0, 1.5, 9, 10, 5};
  for(int i=0; i<points; i++)
  {
    solver.Camera2World(cp[i][0], cp[i][1], cp[i][2],  w2cT, wp[i][0], wp[i][1], wp[i][2]);
    //std::cout << i << "th Cam2World " << wp[i][0] << " " << wp[i][1] << " " << wp[i][2] << std::endl;
    
    double sx, sy, sz;
    solver.World2Sphere(wp[i][0], wp[i][1], wp[i][2],  w2cT, sx, sy, sz);
    //std::cout << i << "th World2Sphere " << sx << " " << sy << " " << sz << std::endl;
  }
  
  // Project camera frame points onto the spherical image
  for(int i=0; i<points; i++)
  {
    solver.Camera2Image(cp[i][0], cp[i][1], cp[i][2], ip[i][0], ip[i][1]);
    //std::cout << i << "th Camera2Image " << ip[i][0] << " " << ip[i][1] << std::endl;
    
    double sx, sy, sz;
    solver.Image2Sphere(ip[i][0], ip[i][1], sx, sy, sz);
    //std::cout << i << "th Image2Sphere " << sx << " " << sy << " " << sz << std::endl;
  }
  
  // Run the solver to check whether it gave correct transform T  
  std::vector<cv::DMatch> matches;
  std::vector<cv::Vec3f> kps3d;
  std::vector<cv::KeyPoint> kps;
  int num_ransac = 7;
  int snp_pairs = 200;
  double threshold = 0.3;
  double solved_w2cT [6] = {0};
  std::vector<int> inliers;
  
  for(int i=0; i<points; i++)
  {
    cv::DMatch m;
    m.queryIdx = i;
    m.trainIdx = i;
    matches.push_back(m);
    kps3d.push_back(cv::Vec3f(wp[i][0], wp[i][1], wp[i][2]));
    kps.push_back(cv::KeyPoint(ip[i][0], ip[i][1], 1));
  }
  solver.Solve(matches, kps3d, kps, snp_pairs, num_ransac, threshold, solved_w2cT, inliers);
  
  // Print out the summary
  std::cout << "Camera position used to generate the synthetic data: (roll, pitch, yaw, x, y, z)\n";
  for(int i=0; i<6; i++){
    std::cout << w2cT[i] << "  ,  ";
  }
  std::cout << std::endl << std::endl;
  
  std::cout << "Camera position estimated by solveSnP: (roll, pitch, yaw, x, y, z)\n";
  for(int i=0; i<6; i++){
    std::cout << solved_w2cT[i] << "  ,  ";
  }
  std::cout << "\nFound inliers " << inliers.size() << " inliers\n";
  
}