#include <Eigen/Core>
#include <boost/concept_check.hpp>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

//#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.h>

class VertexCameraBAL : public g2o::BaseVertex<9,Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::VectorXd::ConstMapType v ( update, VertexCameraBAL::Dimension );
        _estimate += v;
    }

};


class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

class CameraSe3BAL
{
public:
    Sophus::SE3 se3_;
    double f_;
    double k1_;
    double k2_;
};

class VertexCameraSe3BAL : public g2o::BaseVertex<9,CameraSe3BAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraSe3BAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    virtual void oplusImpl ( const double* update )
    {
        Sophus::SE3 up1(Sophus::SO3(update[0],update[1],update[2]),
                        Eigen::Vector3d(update[3], update[4], update[5]) );
        Eigen::Matrix<double, 6,1> update1;
        update1 << update[3], update[4], update[5],
                                update[0],update[1],update[2];
        Sophus::SE3 up = Sophus::SE3::exp(update1);
        _estimate.se3_ = up * _estimate.se3_;
        _estimate.f_ += update[6];
        _estimate.k1_ += update[7];
        _estimate.k2_ += update[8];
    }
};

class EdgeObservationSe3BAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraSe3BAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationSe3BAL() {}
    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }
    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraSe3BAL* cam = static_cast<const VertexCameraSe3BAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        Eigen::Vector3d P = cam->estimate().se3_*(point->estimate()); // position 
        double xp = -P[0]/P[2];
        double yp = -P[1]/P[2];
        const double& f = cam->estimate().f_;
        const double& k1 = cam->estimate().k1_;
        const double& k2 = cam->estimate().k2_;
        double r2 = xp*xp + yp*yp;
        double distortion = 1.0 + r2 * (k1 + k2 * r2);
        _error[0] = _measurement[0] - f * distortion * xp;
        _error[1] = _measurement[1] - f * distortion * yp;
        //_error = -1.0 * _error;
    }
    virtual void linearizeOplus() override
    {
        //BaseBinaryEdge<2, Eigen::Vector2d, VertexCameraSe3BAL, VertexPointBAL>::linearizeOplus();
        //return;
        const VertexCameraSe3BAL* cam = static_cast<const VertexCameraSe3BAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );
        Eigen::Vector3d X = point->estimate();
        Eigen::Vector3d P = cam->estimate().se3_*X; // position 
        double xp = -P[0]/P[2];
        double yp = -P[1]/P[2];
        const double& f = cam->estimate().f_;
        const double& k1 = cam->estimate().k1_;
        const double& k2 = cam->estimate().k2_;
        double n2 = xp*xp + yp*yp;
        double r = 1.0 + n2 * (k1 + k2 * n2);
        Eigen::Vector3d dr_dP;
        Eigen::Vector3d dr_dfk;
        Eigen::Vector3d P2(P[0]*P[0], P[1]*P[1], P[2]*P[2]);
        dr_dP[0] = 2 * k1 * P[0]/P2[2] + k2*4*P[0]*(P2[0]+P2[1])/(P2[2]*P2[2]);
        dr_dP[1] = 2 * k1 * P[1]/P2[2] + k2*4*P[1]*(P2[0]+P2[1])/(P2[2]*P2[2]);
        dr_dP[2] = -2 *k1 * (P2[0]+P2[1])/(P2[2]*P[2]) - 4 * k2 * (P2[0]+P2[1])*(P2[0]+P2[1])/(P2[2]*P2[2]*P[2]);

        Eigen::Matrix<double, 2, 3> duv_dP;  // P is 3d point in camera, P = se3*X;
        Eigen::Matrix<double, 2, 3> duv_dfk; // duv_dfk1k2
        Eigen::Matrix<double, 3, 6> dP_dse3; //se3 is the cam pose
        Eigen::Matrix<double, 2, 3> duv_dX; //X is the 3d point
        Eigen::Matrix<double, 3, 3> dP_dX;
        duv_dP(0,0) = dr_dP[0]*xp + r * (-1) / P[2];
        duv_dP(0,1) = dr_dP[1]*xp;
        duv_dP(0,2) = dr_dP[2]*xp + r * P[0] / P2[2];

        duv_dP(1,0) = dr_dP[0]*yp;
        duv_dP(1,1) = dr_dP[1]*yp + r * (-1) / P[2];
        duv_dP(1,2) = dr_dP[2]*yp + r * P[1] / P2[2];

        duv_dP = f * duv_dP;

        duv_dfk(0,0) = r * xp;
        duv_dfk(0,1) = f * n2 * xp;
        duv_dfk(0,2) = f * n2 * n2 * xp;

        duv_dfk(1,0) = r * yp;
        duv_dfk(1,1) = f * n2 * yp;
        duv_dfk(1,2) = f * n2 * n2 * yp;

        dP_dse3.block<3,3>(0,0) = -Sophus::SO3::hat(P);
        dP_dse3.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<2,6>(0,0) = duv_dP * dP_dse3;
        _jacobianOplusXi.block<2,3>(0,6) = duv_dfk;
        dP_dX = cam->estimate().se3_.rotation_matrix();
        _jacobianOplusXj = duv_dP * dP_dX;
        _jacobianOplusXi = -1 * _jacobianOplusXi;
        _jacobianOplusXj = -1 * _jacobianOplusXj;
    }
};
