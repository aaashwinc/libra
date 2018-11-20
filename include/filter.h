#include <cmath>
#include <teem/nrrd.h>
#include <glm/glm.hpp>
#include <vector>

#ifndef FILTER_H
#define FILTER_H

struct DiscreteKernel{
  int radius;
  int support;
  double *data;
  double *temp;
};

class ScaleBlob{
public:
  std::vector<glm::ivec3> voxels;
  std::vector<ScaleBlob*> children;
  ScaleBlob *parent;

  glm::vec3   position;  // mean of this blob in space.
  glm::mat3x3 shape;     // output of PCA on whole blob.
  glm::vec3   min;       // min bounding box of blob.
  glm::vec3   max;       // max bounding box of blob.
  double      scale;     // scale at which is blob was found.
};

class Filter{
private:
  struct{
    Nrrd  **nrrd;
    short **buff;
    int     nbuf;
    int     curr;
    
    DiscreteKernel kernel;

    NrrdAxisInfo *a;
    int w0,w1,w2,w3,w4;
    int a0,a1,a2,a3;
    bool alive;
  }self;
  int itempbuf(int c);
  int itempbuf();
public:
  Filter();
  void conv1d(Nrrd *nin, Nrrd *nout, int start, int skip, int max, DiscreteKernel kernel);
  void conv2d(short *nin, short *nout, int xmax, int ymax, int zmax, int xstep, int ystep, int zstep, DiscreteKernel kernel);
  void convolve(Nrrd *nin, Nrrd *nout, DiscreteKernel kernel);
  DiscreteKernel gaussian(double sigma, int radius, int d=0);
  DiscreteKernel laplacian();
  void set_kernel(DiscreteKernel k);
  void filter();

  void normalize(double power=1.0);
  void threshold(int min, int max);
  void positive(int channel=0);
  void negative(int channel=0);
  void binary(int channel=0);
  void laplacian3d();
  void median1();
  void maxima();

  std::vector<glm::ivec3> find_maxima();
  void highlight(std::vector<glm::ivec3> points);
  std::vector<ScaleBlob*> find_blobs();

  void init(Nrrd *nin);
  Nrrd *commit();
  void destroy();

  static void print_kernel(DiscreteKernel k);
};

#endif