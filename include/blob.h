#ifndef BLOB_H
#define BLOB_H

#include <glm/glm.hpp>
#include <vector>
using namespace glm;

class ScaleBlob{
public:
  std::vector<ScaleBlob*> children;
  ScaleBlob *parent;

  vec3   mode;      // the local maxima which seeded this blob.
  dvec3  position;  // mean of this blob in space.
  dmat3x3 shape;    // covariance matrix of blob.
  // dmat3x3  eigs; // eigenvectors of covariance matrix.
  mat3x3 invCov;    // inverse of covariance matrix.
  float  detCov;    // determinant of covariance matrix.
  float  pdfCoef;   // |2*pi*covariance_matrix|^(-0.5)
  vec3   min;       // min bounding box of blob.
  vec3   max;       // max bounding box of blob.
  float scale;      // scale at which is blob was found.

  // double volume;
  float n;
  int npass;
  ScaleBlob();
  float pdf(vec3 p);
  float cellpdf(vec3 p);
  void pass(vec3 point, float value);
  void commit();
  void print();
  void printtree(int depth=0);
};

#endif