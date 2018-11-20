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

  Filter gaussian;
  gaussian.set_kernel(gaussian.gaussian(1.0, 3));
  gaussian.init(exp->get(low));
  tick("init.");

  // gaussian.threshold(1000,30000);
  // tick("threshold.");

  // // gaussian.median1();
  // // tick();
  // // return;

  // gaussian.filter();
  // tick("gaussian.");

  // gaussian.laplacian3d();
  // tick("laplacian.");

  // std::vector<glm::ivec3> maxima = gaussian.find_maxima();
  // gaussian.highlight(maxima);
  // tick("highlights.");

  // // printf("normalize\n");
  // // gaussian.normalize(0);
  // // tick();


  // gaussian.maxima();
  // tick("maxima.");

  gaussian.normalize();
  tick("normalize.");

  std::vector<ScaleBlob*> blobs = gaussian.find_blobs();
  tick("blobs");

  gaussian.commit();
  gaussian.destroy();

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