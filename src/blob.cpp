#include "blob.h"
#include <cstdio>



ScaleBlob::ScaleBlob(){

  position = dvec3(0);
  
  shape = mat3x3(0);
  eigs  = mat3x3(0);
  min   = dvec3(0);
  max   = dvec3(0);

  parent = 0;
  scale  = 0;
  volume = 0;
  n      = 0;
  npass  = 0;
}
void ScaleBlob::pass(dvec3 point, double value){
  if(npass == 0){  // calculate mean, min, max.
    position += point*value;
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
    dvec3 v = point - position;
    shape[0][0] += v.x*v.x*value/n;
    shape[0][1] += v.x*v.y*value/n;
    shape[0][2] += v.x*v.z*value/n;
    shape[1][1] += v.y*v.y*value/n;
    shape[1][2] += v.y*v.z*value/n;
    shape[2][2] += v.z*v.z*value/n;
  }
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
  }
}
void ScaleBlob::print(){
  printf("blob at %.2f %.2f %.2f; xyz %.3f %.3f %.3f; xy/xz/yz %.3f %.3f %.3f\n",
    position[0],position[1],position[2],
    shape[0][0],shape[1][1],shape[2][2],
    shape[0][1],shape[0][2],shape[1][2]);
}