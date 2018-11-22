#include "pipeline.h"
#include <chrono> 
#include <iostream>
#include <glm/glm.hpp>
using namespace std::chrono; 
using namespace std;

Pipeline::Pipeline(Experiment *exp){
  this->exp = exp;
}
void Pipeline::init(){
  // gaussian.set_kernel(gaussian.gaussian(3.0, 3, 2));
  // laplacian.set_kernel(gaussian.laplacian());
}

static void tick(std::string text){
  static auto clock = high_resolution_clock::now();
  auto next = high_resolution_clock::now();
  long d = duration_cast<milliseconds>(next-clock).count();
  printf("%s\n",text.c_str());
  printf(">> elapsed %lums\n\n",d);
  clock = next;
}

std::vector<glm::ivec3> find_maxima(Nrrd *in){
  // for(int i=0;i<)
  return std::vector<glm::ivec3>();
}

void Pipeline::process(int low, int high){

  tick("begin.");

  Filter filter;
  filter.init(exp->get(low));
  tick("init.");

  DiscreteKernel gaussian    = filter.gaussian(5.0, 30);
  DiscreteKernel gaussian2   = filter.gaussian(2.0,  10);
  DiscreteKernel interpolate = filter.interpolation();


  // filter.threshold(1000,30000);
  // tick("threshold.");

  // filter.set_kernel(interpolate);
  // filter.filter();




  // filter.median1();
  // tick("median.");
  // filter.median1();
  // tick("median.");
  // // return;

  filter.set_kernel(gaussian);
  filter.filter();
  // filter.filter();
  // filter.normalize(1);
  // filter.filter();
  tick("gaussian.");

  // filter.max1();
  // tick("max filter.");

  // filter.normalize();
  // filter.laplacian3d(1);
  // filter.normalize();
  // filter.set_kernel(gaussian2);
  // filter.filter();
  // filter.print();
  // filter.max1();
  tick("laplacian.");

  // filter.max1();
  // tick("max filter.");

  // std::vector<glm::ivec3> maxima = filter.find_maxima();
  // filter.highlight(maxima);
  // tick("highlights.");

  // // printf("normalize\n");
  // // filter.normalize(0);
  // // tick();


  // filter.maxima();
  // tick("maxima.");

  filter.normalize();
  tick("normalize.");

  std::vector<ScaleBlob*> blobs = filter.find_blobs();
  tick("blobs.");

  // filter.normalize(0.10);
  tick("exponent.");

  filter.commit();
  filter.destroy();

  tick("done.");
  // for(int i=low;i<=high;++i){
  //   ScaleTree tree;
  //   Nrrd *frame = exp.get(i);
  //   std::vector<ScaleBlob> blobs;
  //   copy(frame, st.buf[0]);
  //   for(int i=0;i<10;++i){
  //     gaussian.filter(st.buf[0], st.buf[0], 3.0);
  //     laplacian.filter(st.buf[0],st.buf[1]);
  //     find_next_blobs(st.buf[1],blobs);
  //     // SWAP(st.buf[0],st.buf[1],st.vp);
  //   }
  // }
}