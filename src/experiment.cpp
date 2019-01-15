#include "experiment.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>

static Nrrd* convert_short4_to_float4_and_destroy(Nrrd *nin){
  int a0 = nin->axis[0].size;
  int a1 = nin->axis[1].size;
  int a2 = nin->axis[2].size;
  int a3 = nin->axis[3].size;

  Nrrd *nout = nrrdNew();

  short *in  = (short*)nin->data;
  float *out = new float[a0*a1*a2*a3];

  for(int i=0;i<a0*a1*a2*a3;++i){
    out[i] = (float)(in[i] / 32768.0);
  }

  nrrdWrap_va(nout, out, nrrdTypeFloat, 4, a0, a1, a2, a3);
  nrrdNuke(nin);
  return nout;
}
static Nrrd* load_nrrd(const char *filename){
  printf("load %s\n", filename);
  Nrrd *nrrd = nrrdNew();
  if (nrrdLoad(nrrd, filename, NULL)) {
    char *err = biffGetDone(NRRD);
    fprintf(stderr, "error reading \"%s\":\n%s", filename, err);
    free(err);
    exit(0);
  }
  return convert_short4_to_float4_and_destroy(nrrd);
}


/** Returns length of first sequence of 'c' in str.
  * Eg. f("aabbcccdefccccc", 'c') = 3.
  */ 
static int length_first_repeated_sequence(const char *str, const char c){
  int digits = 0;
  const char *p = str;
  while(*p){          // while not hit null-terminator.
    if(*p == c){      // count number of question marks.    
      ++digits;
    }
    else if(digits){  // no longer a question mark, after
      return digits;  // we started counting (because digits > 0).
    }                 // return digits.
    ++p;
  }
  return digits;
}

ArExperiment::ArExperiment(std::string path, int low, int high, int mem_cap){
  int digits = length_first_repeated_sequence(path.c_str(),'?');
  paths = new std::string[high-low+1];
  char counter[digits+1];
  memset(counter,'\0',digits);
  for(int i=low;i<=high;++i){
    snprintf(counter, sizeof(counter), "%0*d", digits, i);
    std::string filename = path;
    int p;
    while((p=filename.find('?')) != std::string::npos){
      filename = filename.replace(p,digits,counter);
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

Nrrd* ArExperiment::get(int n){
  // printf("get frame %d\n",n);
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

std::string ArExperiment::getfilepath(int n){
  return paths[n-low];
}

Nrrd* ArExperiment::copy(int n){
  Nrrd *src = get(n);
  Nrrd *dst = nrrdNew();

  int a0 = src->axis[0].size;
  int a1 = src->axis[1].size;
  int a2 = src->axis[2].size;
  int a3 = src->axis[3].size;

  float *in  = (float*)src->data;
  float *out = new float[a0*a1*a2*a3];

  for(int i=0;i<a0*a1*a2*a3;++i){
    out[i] = in[i];
  }
  nrrdWrap_va(dst, out, nrrdTypeFloat, 4, a0, a1, a2, a3);
  return dst;
}