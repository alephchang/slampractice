
#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <pangolin/pangolin.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
using namespace std;
void loadTrajectory(std::string path, std::vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3> >& traj0,
    std::vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3> >& traj1)
{
    double word[8];                                                             
    std::ifstream fin(path.c_str());
    if(!fin.good()){
        std::cerr<< "faile to open the trajectory files: "<<path << std::endl;
        return ;
    }
    traj0.clear();
    traj1.clear();
    bool firstTraj=true;
    while(true){
        for(size_t i = 0; i < 8; ++i) fin >> word[i];
        if(fin.eof()) break;
        Eigen::Vector3d translation(word[1],word[2], word[3]);
        Eigen::Quaterniond quaternion(word[7], word[4], word[5], word[6]);
        quaternion.normalize();
        if(firstTraj)
            traj0.push_back(Sophus::SE3(quaternion, translation));
        else
            traj1.push_back(Sophus::SE3(quaternion, translation));
        firstTraj = !firstTraj;
    }
}
void DrawTwoTrajectory(const vector<Eigen::Vector3d> & poses0,
    const vector<Eigen::Vector3d >& poses1) {
    if (poses0.empty() || poses1.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        size_t sz = std::min(poses0.size(), poses1.size());
        for (size_t i = 0; i < sz - 1; i++) {
            glColor3f(1 - (float) i / poses0.size(), 0.0f, (float) i / poses0.size());
            glBegin(GL_LINES);

            auto p1 = poses0[i], p2 = poses0[i + 1];
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3d(p2[0], p2[1], p2[2]);
            
            p1 = poses1[i], p2 = poses1[i + 1];
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3d(p2[0], p2[1], p2[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p, point is p'
        _error = _measurement - pose->estimate().map( _point );
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) { return true;}
    bool write ( ostream& out ) const {return true;}
protected:
    Eigen::Vector3d _point;
};
void icp(const vector<Eigen::Vector3d>& pts1,
    const vector<Eigen::Vector3d>& pts2, Sophus::SE3& se3)
{
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(
                                                         std::unique_ptr<Block>(new Block ( std::unique_ptr<Block::LinearSolverType>(
                                                                    new g2o::LinearSolverEigen<Block::PoseMatrixType>()) )));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId(0);
    pose->setEstimate( g2o::SE3Quat(se3.rotation_matrix(), se3.translation()) );
    optimizer.addVertex( pose );

    // edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(pts2[i]);
        edge->setId( index );
        edge->setVertex( 0, dynamic_cast<g2o::VertexSE3Expmap*> (pose) );
        edge->setMeasurement( pts1[i]);
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
    se3 = Sophus::SE3(pose->estimate().rotation(), pose->estimate().translation());
}

int main()
{
    std::string trajectory_file = "../compare.txt";
    std::vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3> > traj0, traj1;
    loadTrajectory(trajectory_file, traj0, traj1);
    double sum=0.0;
    size_t sz = std::min(traj0.size(), traj1.size());
    std::vector<Eigen::Vector3d> pts0, pts1;
    for(size_t i = 0; i < sz; ++i){
        Sophus::SE3 se3 = traj0[i].inverse()*traj1[i];
        Sophus::Vector6d d = se3.log();
        sum += d.dot(d);
        pts0.push_back(traj0[i].translation());
        pts1.push_back(traj1[i].translation());
    }
    std::cout << "RMSE: " << sqrt(sum/sz) << std::endl;
    
    Sophus::SE3 se3;
    se3 = traj0[0].inverse()*traj1[0];
    icp(pts0, pts1, se3);
    for(size_t i = 0; i < sz; ++i){
        pts1[i] = se3*pts1[i];
    }
    DrawTwoTrajectory(pts0, pts1);
    return 0;
}