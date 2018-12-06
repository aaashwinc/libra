#include "pipeline.h"
#include <chrono> 
#include <iostream>
#include <glm/glm.hpp>
#include <stdio.h>
#include <string.h>
#include <set>
#include <queue>
#include <limits>


using namespace std::chrono; 
using namespace std;

static void tick(std::string text){
  static auto clock = high_resolution_clock::now();
  auto next = high_resolution_clock::now();
  long d = duration_cast<milliseconds>(next-clock).count();
  printf("%s",text.c_str());
  printf(">> elapsed %lums\n\n",d);
  clock = next;
}
ReprMode::ReprMode(const char *name){
  this->name = name;
  blob.scale = 0;
  timestep   = 0;
}
bool ReprMode::operator==(ReprMode &r){
  return (name == r.name) && (timestep == r.timestep)
    && ((blob.scale == r.blob.scale));
}
ArPipeline::ArPipeline(ArExperiment *exp){
  this->exp = exp;
  for(int i=0;i<exp->high-exp->low+1;i++){
    ArFrameData data;
    data.blob     = 0;
    data.complete = false;
    frames.push_back(data);
  }
  store.nbuf = 3;
  store.buf  = new Nrrd*[store.nbuf];
  for(int i=0;i<store.nbuf;++i){
    store.buf[i] = exp->copy(0);
  }
  store.init = true;

  filter.init(store.buf[0]);
}
ArFrameData ArPipeline::get(int frame){
  return frames[frame - exp->low];
}
int ArPipeline::low(){
  return exp->low;
}
int ArPipeline::high(){
  return exp->high;
}

static std::vector<float> collect_scales(ScaleBlob *blob){
  std::set<float> scales;
  std::queue<ScaleBlob*> queue;
  queue.push(blob);
  while(!queue.empty()){
    blob = queue.front();
    queue.pop();
    scales.insert(blob->scale);
    for(ScaleBlob *child : blob->children){
      queue.push(child);
    }
  }
  scales.erase(0);
  return std::vector<float>(scales.begin(), scales.end());
}
static float compute_epsilon(std::vector<float> &v){
  if(v.size() < 2)return 0;
  float eps = fabs(v[1] - v[0]);
  for(int i=1;i<v.size();++i){
    eps = fmin(eps, fabs(v[i]-v[i-1]));
  }
  return eps/2.f;
}

ReprMode ArPipeline::repr_coarser(ReprMode rm){
  std::vector<float> scales = get(rm.timestep).scales;
  ++rm.blob.scale;
  if(rm.blob.scale < 0)rm.blob.scale = 0;
  if(rm.blob.scale >= scales.size()) rm.blob.scale = scales.size()-1;
  return rm;
}
ReprMode ArPipeline::repr_finer(ReprMode rm){
  std::vector<float> scales = get(rm.timestep).scales;
  --rm.blob.scale;
  if(rm.blob.scale < 0)rm.blob.scale = 0;
  if(rm.blob.scale >= scales.size()) rm.blob.scale = scales.size()-1;
  return rm;
}

static std::vector<ScaleBlob*> collect_blobs(ScaleBlob *blob, float scalemin, float scalemax){
  // scalemin = -1000;
  // scalemax = 10000;
  // printf("(");
  std::vector<ScaleBlob*> blobs;
  if(blob->scale >= scalemin || blob->scale == 0){
    if(blob->scale <= scalemax || blob->scale == 0){
      // printf(".");
      blobs.push_back(blob);      
    }
    for(ScaleBlob *child : blob->children){
      std::vector<ScaleBlob*> childblobs = collect_blobs(child, scalemin, scalemax);
      blobs.insert(blobs.end(), childblobs.begin(), childblobs.end());
    }
  }
  // printf(")");
  return blobs;
}

/* store.buf[0] contains the output of the filter chain.
 * store.buf[1] contains the output of repr().
 */
void ArPipeline::process(int low, int high){

  for(int frame=low;frame<=high;frame++){
    printf("pipeline.process %d\n",frame);
    tick("");

    filter.capture(exp->get(frame));

    ScaleBlob *blob             = filter.compute_blob_tree();
    std::vector<float> scales   = collect_scales(blob);

    frames[frame - exp->low].blob      = blob;
    frames[frame - exp->low].scales    = scales;
    frames[frame - exp->low].scale_eps = compute_epsilon(scales);
    frames[frame - exp->low].complete  = true;
    frames[frame - exp->low].bspblobs  = filter.get_bsp(10);;

    BSPTree<ScaleBlob> *bsptree = &frames[frame - exp->low].bspblobs;
    std::vector<ScaleBlob*> allblobs = collect_blobs(blob, 0, std::numeric_limits<float>::infinity());
    
    for(ScaleBlob *sb : allblobs){
      bsptree->insert(sb, sb->position);
    }

    // link with previous frame.
    if(frame-1 >= this->low() && get(frame-1).complete){
      BSPTree<ScaleBlob> *t0 = &frames[ frame - exp->low -1 ].bspblobs;
      BSPTree<ScaleBlob> *t1 = &frames[ frame - exp->low    ].bspblobs;

      std::vector<ScaleBlob*> v0 = t0->as_vector();

      for(ScaleBlob *sb : v0){
        std::vector<ScaleBlob*> potential;
        t1->find_within_distance(potential, sb->position, 200.f);
        for(int i=0;i<potential.size();i++){
          if(sb->distance(potential[i]) < 200.f){
            sb->succ.push_back(potential[i]);            
          }
        }
        // for(ScaleBlob *sb0 : sb->succ){
        //   sb0->pred.push_back(sb);
        // }
      }
    }
    tick("done.\n");
  }
  filter.commit(store.buf[2]);
}

ArGeometry3D* ArPipeline::reprgeometry(ReprMode &mode){
  return &geometry;
}
Nrrd *ArPipeline::repr(ReprMode &mode){
  if(mode.timestep < exp->low)mode.timestep = exp->low;
  if(mode.timestep > exp->high)mode.timestep = exp->high;
  if(!strcmp(mode.name, "plain")){
    // plain representation is simple and is always allowed.
    // default to plain if there the frame has not been processed yet.
    // printf("repr plain\n");
    return exp->get(mode.timestep);
  }
  if(!strcmp(mode.name, "filter residue")){
    printf("repr %s\n", mode.name);
    return store.buf[2];
  }
  if(!strcmp(mode.name, "filter internal")){
    printf("repr %s\n", mode.name);
    return store.buf[0];
  }
  if(!strcmp(mode.name, "sandbox")){
    filter.capture(exp->get(mode.timestep));
    filter.max1();
    // filter.normalize(3.f);
    filter.commit(store.buf[1]);
    return store.buf[1];
  }
  if(!get(mode.timestep).complete){
    return exp->get(mode.timestep);
  }

  static ReprMode last_repr("");
  static Nrrd *last_nrrd;
  if(mode == last_repr){
    // printf("repr %s unchanged.\n",mode.name);
    return last_nrrd;
  }
  last_repr  = mode;
  // printf("repr %s\n",mode.name);
  if(!strcmp(mode.name, "blobs")){
    ArFrameData frame = get(mode.timestep);
    float scalemin = frame.scales[mode.blob.scale] - frame.scale_eps;
    float scalemax = frame.scales[mode.blob.scale] + frame.scale_eps;
    // printf("mode %s. view scale %d with %.2f %.2f\n", mode.name, scalemin, scalemax);
    std::vector<ScaleBlob*> blobs = collect_blobs(frame.blob, scalemin, scalemax);
    // printf("\n");
    filter.clear();
    filter.draw_blobs(blobs, true);
    filter.commit(store.buf[1]);
    return (last_nrrd = store.buf[1]);
  }
  if(!strcmp(mode.name, "gaussian")){
    // printf("mode %s.\n", mode.name);
    ArFrameData frame = get(mode.timestep);
    float scale = frame.scales[mode.blob.scale];
    filter.capture(exp->get(mode.timestep));
    DiscreteKernel kernel = filter.gaussian(scale, int(scale*4));
    filter.set_kernel(kernel);
    filter.max1();
    filter.filter();
    filter.commit(store.buf[1]);
    kernel.destroy();
    return (last_nrrd = store.buf[1]);
  }
  if(!strcmp(mode.name, "laplacian")){
    // printf("mode %s.\n", mode.name);
    ArFrameData frame = get(mode.timestep);
    float scale = frame.scales[mode.blob.scale];
    filter.capture(exp->get(mode.timestep));
    DiscreteKernel kernel = filter.gaussian(scale, int(scale*4));
    filter.set_kernel(kernel);
    filter.max1();
    filter.filter();
    filter.laplacian3d();
    filter.normalize();
    filter.commit(store.buf[1]);
    kernel.destroy();
    return (last_nrrd = store.buf[1]);
  }
}