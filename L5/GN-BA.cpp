//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.h"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "../p3d.txt";
string p2d_file = "../p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    std::ifstream fi(p3d_file.c_str());
    double val[3];
    while(!fi.eof()){
        fi >> val[0] >> val[1] >> val[2];
        p3d.push_back(Eigen::Vector3d(val[0], val[1], val[2]));
        cout << p3d.size() << " " << val[0] << endl;
    }
    fi.close();
    fi.open(p2d_file.c_str());
    while(!fi.eof()){
        fi >> val[0] >> val[1];
        p2d.push_back(Eigen::Vector2d(val[0], val[1]));
    }
    fi.close();
    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;
    Sophus::SE3 T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        Eigen::Vector2d error = Eigen::Vector2d::Zero();
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE 
            Eigen::Vector3d xyz_trans = T_esti*p3d[i];
            double x = xyz_trans[0];
            double y = xyz_trans[1];
            double invz = 1.0/xyz_trans[2];
            double invz_2 = invz*invz;

            Eigen::Vector2d xy_proj(fx*x*invz+cx, fy*y*invz+cy);
            error = p2d[i] - xy_proj;
	    // END YOUR CODE HERE

	    // compute jacobian
            Matrix<double, 2, 6> J;
            J(0,0) = -invz *fx;
            J(0,1) = 0;
            J(0,2) = x*invz_2 *fx;
            J(0,3) =  x*y*invz_2 *fx;
            J(0,4) = -(1+(x*x*invz_2)) *fx;
            J(0,5) = y*invz *fx;

            J(1,0) = 0;
            J(1,1) = -invz *fy;
            J(1,2) = y*invz_2 *fy;
            J(1,3) = (1+y*y*invz_2) *fy;
            J(1,4) = -x*y*invz_2 *fy;
            J(1,5) = -x*invz *fy;

            // START YOUR CODE HERE 

	    // END YOUR CODE HERE

            H += J.transpose() * J;
            b += -J.transpose() * error;
            cost += error.dot(error);
        }

	// solve dx 
        Vector6d dx;

        // START YOUR CODE HERE 
        dx = H.ldlt().solve(b);

        // END YOUR CODE HERE

        if (std::isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE 
        T_esti = Sophus::SE3::exp(dx)*T_esti;
        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
