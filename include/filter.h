#include <cmath>
#include <teem/nrrd.h>
#include <glm/glm.hpp>
#include <vector>
#include "blob.h"

#ifndef FILTER_H
#define FILTER_H

class DiscreteKernel{
public:
  DiscreteKernel();
  ~DiscreteKernel();
  int radius;
  int support;
  double *data;
  double *temp;
  void destroy();
};


class ArFilter{
private:
  struct{
    Nrrd  **nrrd;
    float **buff;   // transient internal buffers for computing
    int     nbuf;   // successive image filters. There are no
    int     curr;   // guarantees about the contents of these images,
                    // except that buff[curr] holds the result of the
                    // most recent filter.
    
    DiscreteKernel kernel;

    NrrdAxisInfo *a;
    int w0,w1,w2,w3,w4;
    int a0,a1,a2,a3;
    bool alive;
  }self;
  int itempbuf(int c);
  int itempbuf();
  double comp_max_laplacian(float *data);
public:
  ArFilter();
  void conv2d(float *nin, float *nout, int xmax, int ymax, int zmax, int xstep, int ystep, int zstep, DiscreteKernel kernel);
  void convolve(Nrrd *nin, Nrrd *nout, DiscreteKernel kernel);
  DiscreteKernel gaussian(double sigma, int radius, int d=0);
  DiscreteKernel interpolation();
  void set_kernel(DiscreteKernel k);
  void filter();

  void normalize(double power=1.0);
  void threshold(int min, int max);
  void positive(int channel=0);
  void negative(int channel=0);
  void binary(int channel=0);
  void laplacian3d(int boundary = 0);
  void max1();
  void median1();
  void maxima();
  void print();
  void clear();

  std::vector<glm::ivec3> find_maxima();
  void highlight(std::vector<glm::ivec3> points);
  std::vector<ScaleBlob*> find_blobs();
  void draw_blobs(std::vector<ScaleBlob*>, bool highlight=false);
  ScaleBlob* compute_blob_tree();

  void capture(Nrrd *nin);
  void init(Nrrd *nin);
  Nrrd *commit(Nrrd *nout = 0);
  void destroy();

  static void print_kernel(DiscreteKernel k);
};

#endif