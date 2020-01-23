#ifndef PIPELINE_H
#define PIPELINE_H

#include <teem/meet.h>
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include <unordered_map>
#include "experiment.h"

#include "shapes.h"
#include "filter.h"
#include "bsptree.h"

#define SWAP(x,y,t) {t=x;x=y;y=t;}

using namespace glm;

class ReprMode{
public:
  ReprMode(const char* name);
  const char *name;
  const char *geom;
  int timestep;
  struct{
    // no parameters here.
  }plain;
  struct{
    int scale;
  }blob;
  struct{
    int timestep;
    vec3 locus;
    std::vector<ScaleBlob*> highlight_loci;
    std::vector<ScaleBlob*>blobs;
    std::vector<vec3> lines;
    std::vector<std::vector<ScaleBlob*>> paths;
    double path_smooth_alpha;
  }highlight;
  bool operator==(ReprMode &r);
};
struct ArFrameData{
  std::string savepath;
  BSPTree<ScaleBlob> bspblobs;
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
  ArGeometry3D geometry;

  std::vector<std::vector<ScaleBlob*>> paths;

public:
  int low();
  int high();
  ArFrameData &get(int frame);

  ArPipeline(ArExperiment *exp);
  void process(int low, int high);
  void find_paths(int minlen, int maxframe, char* mode);
  Nrrd *repr(ReprMode &mode, bool force=false);
  ArGeometry3D *reprgeometry(ReprMode &mode);

  ReprMode repr_coarser(ReprMode);
  ReprMode repr_finer(ReprMode);
  void repr_highlight(ReprMode *rm, vec3 p, vec3 ray, bool diagnose=false, bool add=false);
  void path_highlight(ReprMode *rm, vec3 p, vec3 ray, bool diagnose=false, bool add=false);

  void save();
  void load();
};

#endif