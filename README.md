# solveSnP

Function to estimate camera position from 3D world point - 2D image point correspondences.

Similar to solvePnP function of OpenCV, but for 360 degrees spherical cameras like Ricoh Theta S.

Depends:
- ceres
- OpenCV

Build and demo:
- `$ mkdir build`
- `$ cd build`
- `$ cmake ..`
- `$ make`
- `$ ./demo`


Author:
Manash Pratim Das (mpdmanash@iitkgp.ac.in)
