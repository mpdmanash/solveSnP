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

#include "solveSnP.h"

// Construct Non-Linear Error Function for cere-solver
struct SphericalReprojectionError 
{
  SphericalReprojectionError(double observed_x, double observed_y, double observerd_z, double world_x, double world_y, double world_z)
      : observed_x(observed_x), observed_y(observed_y), observed_z(observed_z), world_x(world_x), world_y(world_y), world_z(world_z)  {}

  template <typename T>
  bool operator()(
          const T* const roll,
          const T* const pitch,
          const T* const yaw,
          const T* const cx,
          const T* const cy,
          const T* const cz,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotations.
    T pc[3];
    const T pw[3] = {T(world_x), T(world_y), T(world_z)};
    const T camera[6] = {roll[0], pitch[0], yaw[0], cx[0], cy[0], cz[0]};
    ceres::AngleAxisRotatePoint(camera, pw, pc);
    // camera[3,4,5] are the translation.
    pc[0] -= cx[0]; pc[1] -= cy[0]; pc[2] -= cz[0];

    T ps[3] = {T(0.0)};
    const T& rho = ceres::sqrt((pc[0]*pc[0])+
                   (pc[1]*pc[1])+
                   (pc[2]*pc[2])
                  );
    ps[0] = pc[0]/rho;
    ps[1] = pc[1]/rho;
    ps[2] = pc[2]/rho;
    
    residuals[0] = ps[0]-T(observed_x);
    residuals[1] = ps[1]-T(observed_y);
    residuals[2] = ps[2]-T(observed_z);
    return true;
  }
private:

  const double observed_x;
  const double observed_y;
  const double observed_z;
  const double world_x;
  const double world_y;
  const double world_z;
};

SolveSnP::SolveSnP(int image_height)
{
    _image_height = image_height;
    _image_center = image_height/2;
    _scale_factor = (double)(image_height)/M_PI;
}

SolveSnP::~SolveSnP()
{
}

void SolveSnP::Solve(std::vector<cv::DMatch> & in_matches, std::vector<cv::Vec3f> & in_kps3d, std::vector<cv::KeyPoint> & in_kps,
                     int snp_pairs, int num_ransac, double threshold, double * final_camera, std::vector<int> & out_inliers)
{
  int max_inliers = 0;
  std::vector<int> final_in_ids;
  
   int threads = omp_get_num_threads();
   std::cout << threads << " threads are being used\n";
  #pragma omp parallel for num_threads(7) shared(final_in_ids, max_inliers)
  for(int i = 0; i < num_ransac; i++)
  {
    double (*world_obs)[3] = new double[snp_pairs][3];
    double (*s_obs)[3] = new double[snp_pairs][3];
    double solved_camera[6];
    
    //Take snp_pairs random match pairs
    int pt_c = 0;
    while(pt_c < snp_pairs){
      int id = rand() % in_matches.size();
      this->Image2Sphere(in_kps[in_matches[id].queryIdx].pt.x, in_kps[in_matches[id].queryIdx].pt.y,
                                s_obs[pt_c][0], s_obs[pt_c][1], s_obs[pt_c][2]);
      world_obs[pt_c][0] = in_kps3d[in_matches[id].trainIdx][0];
      world_obs[pt_c][1] = in_kps3d[in_matches[id].trainIdx][1];
      world_obs[pt_c][2] = in_kps3d[in_matches[id].trainIdx][2];
      pt_c++;
    }
    
    this->_RunOptimizer(snp_pairs, s_obs, world_obs, solved_camera);
    
    int num_inliers;
    std::vector<int> run_in_ids;    
    this->_FindInliers(in_matches, in_kps3d, in_kps, solved_camera, threshold, num_inliers, run_in_ids);
    
    if(num_inliers > max_inliers){
      max_inliers = num_inliers;
      final_in_ids = run_in_ids;
      std::copy(solved_camera, solved_camera+6, final_camera);
      //std::cout << "Interim num-inliers: " << num_inliers << " in " << i << std::endl;
    }
    delete [] world_obs;
    delete [] s_obs;
  }
  out_inliers = final_in_ids;  
}

void SolveSnP::Camera2World(double cx, double cy, double cz, double * w2cT, double & wx, double & wy, double & wz)
{
  double pc[3] = {cx, cy, cz}; // 3D points in camera frame
  double pw[3], pwr[3]; // 3D points in world frame
  double c2wT[6] = {-w2cT[0], -w2cT[1], -w2cT[2], -w2cT[3], -w2cT[4], -w2cT[5]};
  pw[0] = pc[0] - c2wT[3]; pw[1] = pc[1] - c2wT[4]; pw[2] = pc[2] - c2wT[5];  
  ceres::AngleAxisRotatePoint(c2wT, pw, pwr);
  wx = pwr[0];
  wy = pwr[1];
  wz = pwr[2];
}

void SolveSnP::Camera2Image(double cx, double cy, double cz, int & ix, int & iy)
{
  double r = std::sqrt(cx*cx + cy*cy + cz*cz);
  cx = cx/r; cy = cy/r; cz = cz/r;
  double phi = std::asin(cy);
  double theta = std::asin(cx/std::cos(phi));
  iy = phi*_scale_factor + _image_center;
  ix = theta*_scale_factor + _image_center;
  if(cz<0)
  {
    ix = _image_height*2 - 1 - ix;
  }
}

void SolveSnP::Image2Sphere(int x, int y, double & sx, double & sy, double & sz)
{
  double mod_obs_x = x;
  if(x >= _image_center*2)
    mod_obs_x = _image_height*2 - x - 1;
  double phi = (y - _image_center)/_scale_factor;
  double theta = (mod_obs_x - _image_center)/_scale_factor;
  double obs_ps[3] = {0.0};
  obs_ps[0] = std::sin(theta) * std::cos(phi);
  obs_ps[1] = std::sin(phi);
  obs_ps[2] = std::cos(theta) * std::cos(phi);
  if((x >= _image_center*2 && obs_ps[2] >= 0) || // Point on right side should have neg z
      (x < _image_center*2 && obs_ps[2] < 0)){ // Point on left side should have pos z
    //obs_ps[0] = -obs_ps[0];
    //obs_ps[1] = -obs_ps[1];
    obs_ps[2] = -obs_ps[2];
  }
  double rho = std::sqrt((obs_ps[0]*obs_ps[0])+
                         (obs_ps[1]*obs_ps[1])+
                         (obs_ps[2]*obs_ps[2]));
  // 3D points projected on the sphere
  sx = obs_ps[0]/rho;
  sy = obs_ps[1]/rho;
  sz = obs_ps[2]/rho;
}

void SolveSnP::World2Sphere(double x, double y, double z, double * w2cT, double & sx, double & sy, double & sz)
{
  double pc[3]; // 3D points in camera frame
  double pw[3] = {x, y, z}; // 3D points in world frame
  ceres::AngleAxisRotatePoint(w2cT, pw, pc);
  pc[0] -= w2cT[3]; pc[1] -= w2cT[4]; pc[2] -= w2cT[5];
  double rho = std::sqrt((pc[0]*pc[0])+
                         (pc[1]*pc[1])+
                         (pc[2]*pc[2]));
  // 3D points projected on the sphere
  sx = pc[0]/rho;
  sy = pc[1]/rho;
  sz = pc[2]/rho;
}

void SolveSnP::_RunOptimizer(int num_pairs, double s_obs[][3], double world_obs[][3], double * out_cam, bool use_initial)
{
  double roll = 0, pitch = 0, yaw = 0, cx = 0, cy = 0, cz = 0;
  if(use_initial)
  {
    roll = out_cam[0]; pitch = out_cam[1]; yaw = out_cam[2]; cx = out_cam[3]; cy = out_cam[4]; cz = out_cam[5];
  }
  ceres::Problem problem;
  for (int j = 0; j < num_pairs; j++) 
  {
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<SphericalReprojectionError, 3, 1,1,1,1,1,1>(
    new SphericalReprojectionError(s_obs[j][0],
                                   s_obs[j][1],
                                   s_obs[j][2],
                                   world_obs[j][0],
                                   world_obs[j][1],
                                   world_obs[j][2])
      ), NULL,
      &roll, &pitch, &yaw, &cx, &cy, &cz
    );
  }
  problem.SetParameterLowerBound(&roll, 0, -M_PI);
  problem.SetParameterLowerBound(&pitch, 0, -M_PI);
  problem.SetParameterLowerBound(&yaw, 0, -M_PI);
  
  problem.SetParameterUpperBound(&roll, 0, M_PI);
  problem.SetParameterUpperBound(&pitch, 0, M_PI);
  problem.SetParameterUpperBound(&yaw, 0, M_PI);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.FullReport() << "\n";
  
  double solved_camera[6] = {roll, pitch, yaw, cx, cy, cz}; // r, p, y, x, y, z
  std::copy(solved_camera, solved_camera+6, out_cam);
}

void SolveSnP::_FindInliers(std::vector<cv::DMatch> & in_matches, std::vector<cv::Vec3f> & in_kps3d,
                            std::vector<cv::KeyPoint> & in_kps, double * solved_camera, double threshold,
                            int & out_num_inliers, std::vector<int> & out_in_ids)
{  
  out_num_inliers = 0;
  for(int j=0; j<in_matches.size(); j++)
  {
    
    double isx, isy, isz, wsx, wsy, wsz;
    this->Image2Sphere(in_kps[in_matches[j].queryIdx].pt.x, in_kps[in_matches[j].queryIdx].pt.y,
                       isx, isy, isz);
    this->World2Sphere(in_kps3d[in_matches[j].trainIdx][0], in_kps3d[in_matches[j].trainIdx][1],
                       in_kps3d[in_matches[j].trainIdx][2], solved_camera, wsx, wsy, wsz);
    double distance = std::sqrt( (isx-wsx)*(isx-wsx) +
                                 (isy-wsy)*(isy-wsy) +
                                 (isz-wsz)*(isz-wsz) );
    if(distance < threshold)
    { // Inlier
      out_in_ids.push_back(j);
      out_num_inliers++;
    }
  }
}
