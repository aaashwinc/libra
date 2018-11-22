#ifndef BLOB_H
#define BLOB_H

#include <glm/glm.hpp>
#include <vector>
using namespace glm;

class ScaleBlob{
public:
  std::vector<ScaleBlob*> children;
  ScaleBlob *parent;

  dvec3   position;  // mean of this blob in space.
  mat3x3  shape;     // covariance matrix of blob.
  mat3x3  eigs;      // eigenvectors of covariance matrix.
  dvec3   min;       // min bounding box of blob.
  dvec3   max;       // max bounding box of blob.
  double scale;      // scale at which is blob was found.

  double volume;
  int n;
  int npass;
  ScaleBlob();
  void pass(dvec3 point, double value);
  void commit();
  void print();
};

#endif