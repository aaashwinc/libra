#include "blob.h"
#include <cstdio>
#include <cmath>
#include <limits>

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

    invCov    = glm::inverse(mat3(shape));
    detCov    = glm::determinant(shape);
    pdfCoef   = pow(glm::determinant(shape*pi*2.0),-0.5);
  }
}
float ScaleBlob::pdf(vec3 p){
  p = p - vec3(position);
  return pdfCoef * exp(-0.5 * glm::dot(p,(invCov*p)));
}
float ScaleBlob::cellpdf(vec3 p){
  p = p - vec3(position);
  float mag = glm::dot(p,(invCov*p));
  return 1.f/(0.1f + 0.05f*mag*mag);
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

    covariance << shape[0][0], shape[0][1], shape[0][2],
                  shape[1][0], shape[1][1], shape[1][2],
                  shape[2][0], shape[2][1], shape[2][2];
  }
}
void ScaleBlob::print(){
  printf("blob at %.2f %.2f %.2f; xyz %.3f %.3f %.3f; xy/xz/yz %.3f %.3f %.3f\n",
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

// compute Wasserstein distance: https://en.wikipedia.org/wiki/Wasserstein_metric
float ScaleBlob::distance(ScaleBlob *blob){
  using namespace Eigen;
  SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  
  Matrix3f sqc2 = solver.operatorSqrt();
  Matrix3f c2c1c2 = sqc2 * blob->covariance * sqc2;
  
  solver = SelfAdjointEigenSolver<Matrix3f>(c2c1c2);
  Matrix3f sqrtc2c1c2 = solver.operatorSqrt();
  Matrix3f whole = blob->covariance + covariance - (2.f * sqrtc2c1c2);

  float trace = whole.trace();
  vec3  delta = blob->position - position;
  return (dot(delta, delta)) + trace;

  // return length(blob->position - position);
}