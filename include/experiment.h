#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <teem/nrrd.h>
#include <string>
#include <map>

struct NrrdFrame{
  int n;
  long accessed;
  std::string path;
  Nrrd* nrrd;
};
class Experiment{
public:
  Experiment(std::string path, int digits, int low, int high, int mem_cap);
  NrrdFrame* frames;
  Nrrd* get(int n);
  int low;
  int high;
private:
  std::string *paths;
  int nframes;
  int npaths;
  long time;
};

#endif