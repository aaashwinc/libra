#ifndef PIPELINE_H
#define PIPELINE_H

#include <teem/meet.h>
#include <vector>
#include <glm/glm.hpp>
#include "experiment.h"
#include "filter.h"

#define SWAP(x,y,t) {t=x;x=y;y=t;}

using namespace glm;


class ScaleTree{
  ScaleBlob root;
};
class Pipeline{
  typedef void (*fdatacb)(Nrrd*);
private:
  fdatacb callback;
  Experiment *exp;
  struct Frame{
    std::vector<ivec2> maxima;
  };
  struct{
    Nrrd **buf; // buffers to store intermediate Nrrds
    void *vp;   // temporary pointer, useful for swapping
    int nbuf;   // number of buffers stored
    bool init;  // initialized?
  }st;
  Filter gaussian;
  Filter laplacian;
public:
  Pipeline(Experiment *exp);
  void set_callback(fdatacb cb);
  void set_experiment(Experiment *exp);
  std::vector<ivec3> hill_climb(Nrrd *nin, std::vector<ivec3> start);
  void find_next_blobs(Nrrd *nin, std::vector<ScaleBlob> blobs);
  void copy(Nrrd *nin, Nrrd *nout);
  void init();
  void process(int low, int high);
};

#endif