#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <teem/nrrd.h>
#include <string>
#include <map>
#include "filter.h"

struct NrrdFrame{
  int n;
  long accessed;
  std::string path;
  Nrrd* nrrd;
};

// Exposes methods for interacting with
// the raw data of the experiment.
class ArExperiment{
public:
  ArExperiment(std::string path, int low, int high, int mem_cap);
  NrrdFrame* frames;
  Nrrd* get(int n);
  Nrrd* copy(int n);
  int low;
  int high;
private:
  ArFilter filter;
  std::string *paths;
  int nframes;
  int npaths;
  long time;
};

#endif