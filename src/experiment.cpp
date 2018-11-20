#include "experiment.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>

// nrrdNuke(nrrds[n]);

static Nrrd* load_nrrd(const char *filename){
  printf("load %s\n", filename);
  Nrrd *nrrd = nrrdNew();
  if (nrrdLoad(nrrd, filename, NULL)) {
    char *err = biffGetDone(NRRD);
    fprintf(stderr, "error reading \"%s\":\n%s", filename, err);
    free(err);
    exit(0);
  }
  return nrrd;
}


Experiment::Experiment(std::string path, int digits, int low, int high, int mem_cap){
  paths = new std::string[high-low+1];
  char counter[digits+1];
  memset(counter,'\0',digits);
  for(int i=low;i<=high;++i){
    snprintf(counter, sizeof(counter), "%0*d", digits, i);
    std::string filename = path;
    int p;
    while((p=filename.find('?')) != std::string::npos){
      filename = filename.replace(p,1,counter);
    }
    paths[i] = filename;
  }

  frames = new NrrdFrame[mem_cap];
  for(int i=0;i<mem_cap;++i){
    frames[i].n        = -1;
    frames[i].accessed = 0;
    frames[i].path     = "";
    frames[i].nrrd     = 0;
  }
  this->low     = low;
  this->high     = high;
  this->nframes = mem_cap;
  this->npaths  = (high-low+1);
  this->time    = 0;
}

Nrrd* Experiment::get(int n){
  printf("get frame %d\n",n);
  ++time;
  int min_i   = 0;
  int min_acc = frames[0].accessed;
  // look for frame n in memory.
  for(int i=0;i<nframes;++i){
    if(frames[i].accessed < min_acc){
      min_i = i;
      min_acc = frames[i].accessed;
    }
    if(frames[i].n == n){
      frames[i].accessed = time;
      return frames[i].nrrd;
    }
  }
  // frame n not in memory. load from disk.
  int i=min_i;
  if(frames[i].nrrd){
    nrrdNuke(frames[i].nrrd);
  }
  // printf("loading %s\n",paths[n-low]);
  frames[i].n = n;
  frames[i].accessed = time;
  frames[i].path = paths[n-low];
  frames[i].nrrd = load_nrrd(paths[n-low].c_str());

  filter.init(frames[i].nrrd);
  filter.normalize();
  filter.commit();

  // printf("loaded frame %d: %u, %s, %p\n", n, time, paths[n-low].c_str(), frames[i].nrrd);
  return frames[i].nrrd;
}