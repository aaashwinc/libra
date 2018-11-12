#include "experiment.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>

Experiment::Experiment(std::string path){
  nrrds = new Nrrd*[1];
  for(int i=0;i<1;++i){
    nrrds[i] = 0;
  }
}
void Experiment::unload_nnrd(int n){
  nrrdNuke(nrrds[n]);
  nrrds[n] = 0;
}
Nrrd* Experiment::load_nrrd(const char *filename){
  printf("load %s\n", filename);
  Nrrd *nrrd = nrrdNew();
  if (nrrdLoad(nrrd, filename, NULL)) {
    char *err = biffGetDone(NRRD);
    fprintf(stderr, "error reading \"%s\":\n%s", filename, err);
    free(err);
    exit(0);
  }
  // printf("axis0 %d\n", nrrd->axis[0].size);
  return nrrd;
}
Nrrd* Experiment::load_nrrd(std::string path, int digits, int i){
  char counter[digits+1];
  memset(counter,'\0',digits);
  snprintf(counter, sizeof(counter), "%0*d", digits, i);
  std::string filename = path;
  int p;
  while((p=filename.find('?')) != std::string::npos){
    filename = filename.replace(p,1,counter);
  }
  unload_nrrd(0);
  nrrds[0] = load_nrrd(filename.c_str());
}