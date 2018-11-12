#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <teem/nrrd.h>
#include <string>
#include <map>


class Experiment{
public:
  Nrrd** nrrds;
  Experiment(std::string path, int digits, int low, int high);
  Nrrd* load_nrrd(std::string *path, int digits, int n);
private:
  
  Nrrd* load_nrrd(const char *filename);
};

#endif