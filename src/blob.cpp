#include "blob.h"
#include <cstdio>
#include <cmath>
#include <limits>
#include <LBFGS.h>
#include <iostream>
extern "C"{
  #include <lbfgs.h>
}
#define pi (3.14159265358979323846264338)

ScaleBlob::ScaleBlob(){

  position = vec3(0);
  
  shape = mat3x3(0);
  // eigs  = mat3x3(0);
  float inf = std::numeric_limits<float>::infinity();
  min   = vec3(inf, inf, inf);
  max   = vec3(0);

  parent = 0;
  scale  = 0;
  // volume = 0;
  n      = 0;
  npass  = 0;

  pred = std::vector<ScaleBlob*>();
  succ = std::vector<ScaleBlob*>();

  children = std::vector<ScaleBlob*>();
}
void ScaleBlob::pass(vec3 point, float value){
  if(npass == 0){  // calculate mean, min, max.
    position += dvec3(point*value);
    n += value;
    if(point.x<min.x)min.x = point.x;
    if(point.y<min.y)min.y = point.y;
    if(point.z<min.z)min.z = point.z;
    if(point.x>max.x)max.x = point.x;
    if(point.y>max.y)max.y = point.y;
    if(point.z>max.z)max.z = point.z; 
  }
  if(npass == 1){  // calculate covariance.
    // printf("yo!");
    vec3 v = point - vec3(position);
    shape[0][0] += v.x*v.x*value/(n-1);
    shape[0][1] += v.x*v.y*value/(n-1);
    shape[0][2] += v.x*v.z*value/(n-1);
    shape[1][1] += v.y*v.y*value/(n-1);
    shape[1][2] += v.y*v.z*value/(n-1);
    shape[2][2] += v.z*v.z*value/(n-1);


  }
}
float ScaleBlob::pdf(vec3 p){
  p = p - vec3(position);
  return pdfCoef * exp(-0.5 * glm::dot(p,(invCov*p)));
}
float ScaleBlob::cellpdf(vec3 p){
  p = p - vec3(position);
  float mag = glm::dot(p,(invCov*p));
  if(mag<1.f) return 1.f;
  else        return float(erf(2-mag)*0.5+0.5);
  // float mag = glm::dot(p,(invCov*p));
  // return 1.f/(0.1f + 0.05f*mag*mag);
}
void ScaleBlob::commit(){
  if(npass == 0){  // compute mean.
    position /= double(n);
    npass = 1;
  }
  else if(npass == 1){  // compute covariance matrix.
    // shape /= double(n-1);
    shape[1][0] = shape[0][1];
    shape[2][0] = shape[0][2];
    shape[2][1] = shape[1][2];
    npass = 2;

    invCov    = glm::inverse(mat3(shape));
    detCov    = fabs(glm::determinant(shape));
    pdfCoef   = pow(glm::determinant(shape*pi*2.0),-0.5);

    fshape = shape;

    covariance << invCov[0][0], invCov[0][1], invCov[0][2],
                  invCov[1][0], invCov[1][1], invCov[1][2],
                  invCov[2][0], invCov[2][1], invCov[2][2];
  }
}
void ScaleBlob::print(){
  printf("blob %.2f at %.2f %.2f %.2f; xyz %.3f %.3f %.3f; xy/xz/yz %.3f %.3f %.3f\n", n,
    position[0],position[1],position[2],
    shape[0][0],shape[1][1],shape[2][2],
    shape[0][1],shape[0][2],shape[1][2]);
}
void ScaleBlob::printtree(int depth){
  // printf("\n");
  // for(int i=0;i<depth;++i)printf(" ");
  printf("(%.f",scale);
  for(auto c : children){
    c->printtree(depth+1);
  }
  printf(")");
} 
// float ScaleBlob::distance(vec3 p){
//   // = distance between (p, shape) and position
//   // = distance between (c, shape) and origin.
//   // vec3 c  = vec3(position) - p;
//   // vec3 x  = c + shape*vec3(0,0,1); // inital point x on ellipsoid.
//   // vec3 a(1,0,0);
//   return 0
// }

using namespace Eigen;
static inline float ell(const Matrix3f &A, const Vector3f &x){
  float c00 = x[0]*x[0]*A(0,0);
  float c01 = x[0]*x[1]*A(0,1);
  float c02 = x[0]*x[2]*A(0,2);
  float c11 = x[1]*x[1]*A(1,1);
  float c12 = x[1]*x[2]*A(1,2);
  float c22 = x[2]*x[2]*A(2,2);

  return c00 + c11 + c22 + 2*c01 + 2*c02 + 2*c12;
}
float ScaleBlob::distance(ScaleBlob *blob){
  // printf("d");
  // printf("d...");
  /*** intersection distance between ellipsoids ***/
  using namespace Eigen;

  // define objective function.
  class EllipsoidDistanceObjective{
  public:
    EllipsoidDistanceObjective(const Matrix3f &A, const Vector3f p): A(A), p(p) {}
    Matrix3f A;
    Vector3f p;
    float operator()(const VectorXf &x, VectorXf &grad){
      float xTAx = ell(A, x);
      float xAx = ell(A, x);
      float xTx  = x.dot(x);
      float xp  = x.dot(p);
      float VxAx = sqrt(xAx);
      Vector3f Ax  = A*x;
      Vector3f xTAAT = (x.transpose() * (A + A.transpose())).transpose();

      grad = (2.f*x/(xAx)) + (-xTx/(xAx*xAx) * xTAAT) - (((1.f/VxAx)*2*p) + (2*xp)*(-xTAAT/(2.f*(VxAx*VxAx*VxAx))));
      // printf("] VxAx   = %.30f\n", VxAx);
      // printf("] p      = %.30f %.30f %.30f\n", p[0], p[1], p[2]);
      // printf("] 2/VxAx = %.30f\n", (1.f/VxAx)*2.f);
      // std::cout << "compute:\n"
      //   << (2.f*x/(xAx)) << "\n"
      //   << (-xTx/(xAx*xAx) * xTAAT) << "\n"
      //   << ((1.f/VxAx)*2.f*p) << "\n"
      //   << ((2.f*xp)*(-xTAAT/(2.f*(VxAx*VxAx*VxAx)))) << "\n";
      return (1.f/xAx * xTx) - (1.f/sqrt(xAx))*2.f*xp;
    }
    // map R^3 -> surface of ellipsoid.
    static inline Vector3f g(Matrix3f &A, const VectorXf &x){
      return x*(1.f/sqrt(ell(A,x)));
    }
  };

  // perform precomputation.
  typedef SelfAdjointEigenSolver<Eigen::Matrix3f> Solver3f;
  Vector3f p0(0, 0, 0);


  Solver3f s_A0(covariance);
  Solver3f s_A1(blob->covariance);

  Matrix3f VA0 = s_A0.operatorSqrt();
  Matrix3f VA1 = s_A1.operatorSqrt();

  Matrix3f VA0i = VA0.inverse();
  Matrix3f VA1i = VA1.inverse();

  Matrix3f VA0A1i  = VA0 * VA1i;
  Matrix3f VA1A0iv = VA1 * VA0i;
  
  // std::cout << "VA0 = \n" << VA0 << "\n\n";
  // std::cout << "VA1 = \n" << VA1 << "\n\n";

  // std::cout << "VA0i = \n" << VA0i << "\n\n";
  // std::cout << "VA1i = \n" << VA1i << "\n\n";

  // std::cout << "VA0A1i = \n" << VA0A1i  << "\n\n";
  // std::cout << "VA1A0i = \n" << VA1A0iv << "\n\n";

  Vector3f p1 = VA0 * Vector3f( position[0]-blob->position[0],
                                position[1]-blob->position[1],
                                position[2]-blob->position[2]) ;

  VectorXf x  = EllipsoidDistanceObjective::g(VA0A1i, p1);
  
  Matrix3f Aq = VA1A0iv * VA1A0iv;
  Matrix3f As;
  As <<   Aq(0,0),               (Aq(1,0)+Aq(0,1))/2.f, (Aq(2,0)+Aq(0,2))/2.f,
          (Aq(1,0)+Aq(0,1))/2.f, Aq(1,1),               (Aq(2,1)+Aq(1,2))/2.f,
          (Aq(2,0)+Aq(0,2))/2.f, (Aq(2,1)+Aq(1,2))/2.f, Aq(2,2);

  // printf("\n");
  // printf("p0 = %.3f, %.3f, %.3f\n", position.x, position.y, position.z);
  // printf("p1 = %.3f, %.3f, %.3f\n", blob->position.x, blob->position.y, blob->position.z);
  // std::cout << "A0   = \n" << covariance       << "\n\n";
  // std::cout << "A1   = \n" << blob->covariance << "\n\n";


  // std::cout << "p1   = \n" << p1 << "\n\n";
  // std::cout << "x    = \n" << x  << "\n\n";
  // std::cout << "Aq   = \n" << Aq  << "\n\n";
  // std::cout << "As   = \n" << As  << "\n\n";

  LBFGSpp::LBFGSParam<float> param;
  param.past  = 2;
  param.delta = 0.00001f;
  LBFGSpp::LBFGSSolver<float> solver(param);

  float fx;

  // printf("minimizing %f %f.\n", n, blob->n);
  EllipsoidDistanceObjective obj(As, p1);
  // printf("v");
  int nitr = solver.minimize(obj, x, fx);

  x = EllipsoidDistanceObjective::g(As, x);
  float distance = (x - p1).norm();
  // if(distance < 1.f){
  //   // printf("(%.3f %.3f %.3f) - (%.3f %.3f %.3f)\nd=%.2f\n", position[0], position[1], position[2], blob->position[0], blob->position[1], blob->position[2], distance);    
  // }
  // printf(".../d\n");
  return distance;  


  // compute distance between ellipsoids using newton's method.

  // vec3 p;


  /*** My distance: ***/
  // return (glm::length(blob->position - position)) + fabs(cbrt(detCov) - cbrt(blob->detCov));

  /*** Wasserstein metric:  ***/

  // using namespace Eigen;
  // SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  
  // Matrix3f sqc2 = solver.operatorSqrt();
  // Matrix3f c2c1c2 = sqc2 * blob->covariance * sqc2;
  
  // solver = SelfAdjointEigenSolver<Matrix3f>(c2c1c2);
  // Matrix3f sqrtc2c1c2 = solver.operatorSqrt();
  // Matrix3f whole = blob->covariance + covariance - (2.f * sqrtc2c1c2);

  // float trace = whole.trace();
  // vec3  delta = blob->position - position;
  // return (dot(delta, delta)) + trace;


  /*** simple distance ***/
  // return length(blob->position - position);
}
