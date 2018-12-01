#ifndef PIPELINE_H
#define PIPELINE_H

#include <teem/meet.h>
#include <vector>
#include <glm/glm.hpp>
#include "experiment.h"
#include "filter.h"

#define SWAP(x,y,t) {t=x;x=y;y=t;}

using namespace glm;

class ReprMode{
public:
  ReprMode(const char* name);
  const char *name;
  int timestep;
  struct{
    // no parameters here.
  }plain;
  struct{
    int scale;
  }blob;
  bool operator==(ReprMode &r);
};
struct ArFrameData{
  ScaleBlob *blob;
  std::vector<float> scales;
  float scale_eps;            // some epsilon that satisfies: for all i, scales[i+1] - scales[i] > epsilon.
  bool complete;
};
// A ArPipeline performs the entire analysis pipeline,
// for all frames. Input: experiment. Output: tracks.
// Also exposes functions to visualize intermediate
// steps.
typedef void (*fdatacb)(Nrrd*);
class ArPipeline{
private:
  struct{
    Nrrd **buf; // buffers to store intermediate Nrrds
    int nbuf;   // number of buffers stored
    bool init;  // initialized?
  }store;

  ArFilter filter;
  ArExperiment *exp;
  std::vector<ArFrameData> frames;

public:
  int low();
  int high();
  ArFrameData get(int frame);

  ArPipeline(ArExperiment *exp);
  void process(int low, int high);
  Nrrd *repr(ReprMode &mode);

  ReprMode repr_coarser(ReprMode);
  ReprMode repr_finer(ReprMode);
};

#endif