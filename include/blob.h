#ifndef BLOB_H
#define BLOB_H

#include <glm/glm.hpp>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <vector>
using namespace glm;


class ScaleBlob{
public:
  ScaleBlob *parent;                // scale-tree parent
  std::vector<ScaleBlob*> children; // scale-tree children

  std::vector<ScaleBlob*> pred;     // temporal predecessors
  std::vector<ScaleBlob*> succ;     // temporal successors

  vec3   mode;      // the local maxima which seeded this blob.
  dvec3  position;  // mean of this blob in space.
  dmat3x3 shape;    // covariance matrix of blob.
  mat3x3 fshape;    // covariance matrix of blob.
  int timestep;     // the timestep in which this blob exists.

  Eigen::Matrix3f covariance;


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

  // bool initialized;

  ScaleBlob();
  float pdf(vec3 p);
  float cellpdf(vec3 p);
  void pass(vec3 point, float value);
  void commit();
  void print();
  void printtree(int depth=0);
  float distance(ScaleBlob*);
};

#endif