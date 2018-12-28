#include "pipeline.h"
#include <chrono> 
#include <iostream>
#include <glm/glm.hpp>
#include <stdio.h>
#include <string.h>
#include <set>
#include <queue>
#include <limits>
#include <queue>
#include <SFML/Graphics.hpp>

#include "blobmath.h"


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
  this->geom = "none";
  blob.scale = 0;
  timestep   = 0;
  highlight.blobs = std::vector<ScaleBlob*>();
  highlight.lines = std::vector<vec3>();
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
ArFrameData &ArPipeline::get(int frame){
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

void ArPipeline::repr_highlight(ReprMode *rm, vec3 p, vec3 ray, bool add){
  if(get(rm->timestep).complete){
    // printf("highlight along (%.2f %.2f %.2f) + (%.2f %.2f %.2f)\n",p.x,p.y,p.z, ray.x,ray.y,ray.z);
    ray = normalize(ray)*3.f;
    std::vector<ScaleBlob*> found;
    ArFrameData frame = get(rm->timestep);
    BSPTree<ScaleBlob> *bsptree = &(frame.bspblobs);
    for(int i=0;i<100 && found.size()==0;++i){
      bsptree->find_within_distance(found, p, i+1);
      if(found.size()>0){
        found = std::vector<ScaleBlob*>();
        bsptree->find_within_distance(found, p, 900);
      }
      p+=ray;
    }
    if(!add)rm->highlight.blobs.clear();
    for(int i=0;i<found.size();i++)rm->highlight.blobs.push_back(found[i]);
    for(ScaleBlob *b : found){
      b->print();
      // printf("found det=%.3f\n", b->detCov);
    }
    rm->highlight.paths = longest_paths(rm->highlight.blobs);
    // printf("highlight %d\n",found.size());
  }
  // rm->highlight.lines.push_back(p+ray*1.f);
  // rm->highlight.lines.push_back(p+ray*100.f);
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

/* find all paths in the part of the dataset that has been
 * processed so far. 
 *
 */
void ArPipeline::find_paths(){
  int minlen = 0;
  std::vector<ScaleBlob*> allblobs;
  for(int i=0;i<frames.size();i++){
    if(frames[i].complete){
      std::vector<ScaleBlob*> blobsi = collect_blobs(frames[i].blob, 0, std::numeric_limits<float>::infinity());
      allblobs.insert(allblobs.end(), blobsi.begin(), blobsi.end());
      // minlen = int(i*0.75);
    }
  }
  printf("find paths > %d.\n", minlen);
  printf("find paths in %d blobs.\n", allblobs.size());
  std::vector<std::vector<ScaleBlob*>> allpaths = longest_paths(allblobs);
  paths.clear();
  for(int i=0;i<allpaths.size();++i){
    if(allpaths[i].size() > minlen){
      paths.push_back(allpaths[i]);
    }
  }
  printf("found %d paths.\n", paths.size());
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
    // printf("linking.\n");
    if(frame-1 >= this->low() && get(frame-1).complete){
      BSPTree<ScaleBlob> *t0 = &frames[ frame - exp->low -1 ].bspblobs;
      BSPTree<ScaleBlob> *t1 = &frames[ frame - exp->low    ].bspblobs;

      std::vector<ScaleBlob*> v0 = t0->as_vector();
      int itr = 0;
      for(ScaleBlob *sb : v0){
        std::vector<ScaleBlob*> potential;
        t1->find_within_distance(potential, sb->position, 200.f);
        for(int i=0;i<potential.size();i++){
          if(sb->n>1 && potential[i]->n>1 && sb->distance(potential[i]) <= 1.f){
            if(sb->detCov/potential[i]->detCov < 2.f && potential[i]->detCov/sb->detCov < 2.f){
              // volume cannot more than double or half.
              sb->succ.push_back(potential[i]);
              potential[i]->pred.push_back(sb);
            }
          }
          // printf("%d.", i);
        }
        // printf("o");
        ++itr;
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
  geometry.lines = std::vector<vec3>();
  geometry.lines_c = std::vector<sf::Color>();
  // printf("mode %s\n", mode.geom);
  if(!strcmp(mode.geom, "graph") || !strcmp(mode.geom, "paths") || !strcmp(mode.geom, "succs")){
    if(get(mode.timestep).complete){
      ArFrameData frame = get(mode.timestep);
      float scalemin = frame.scales[mode.blob.scale] - frame.scale_eps;
      float scalemax = frame.scales[mode.blob.scale] + frame.scale_eps;
      std::vector<ScaleBlob*> blobs;

      // if(!strcmp(mode.name, "plain")){
      //   blobs = collect_blobs(frame.blob, 0, std::numeric_limits<float>::infinity());
      // }else{
      //   blobs = collect_blobs(frame.blob, scalemin, scalemax);
      // }


      if(!strcmp(mode.geom, "graph")){
        // draw scale hierarchy:
        blobs = collect_blobs(frame.blob, 0, std::numeric_limits<float>::infinity());
        for(ScaleBlob *sb : mode.highlight.blobs){
        // for(ScaleBlob *sb : blobs){
          // for(ScaleBlob *succ : sb->succ){
          //   geometry.lines.push_back(sb->position);
          //   geometry.lines.push_back(succ->position);
          //   geometry.lines_c.push_back(sf::Color(255,255,255,120));
          //   geometry.lines_c.push_back(sf::Color(255,255,255,120));
          // }
          // for(ScaleBlob *pred : sb->pred){
          //   geometry.lines.push_back(sb->position);
          //   geometry.lines.push_back(pred->position);
          //   geometry.lines_c.push_back(sf::Color(80,80,80,80));
          //   geometry.lines_c.push_back(sf::Color(80,80,80,80));
          // }
          if(sb->parent){
            geometry.lines.push_back(sb->position);
            geometry.lines.push_back(sb->parent->position);
            geometry.lines_c.push_back(sf::Color(0,0,255,150));
            geometry.lines_c.push_back(sf::Color(0,0,255,150));
          }
        }
      }
      // draw successors and predecessors:
      if(!strcmp(mode.geom, "succs")){
        float scalemin = frame.scales[mode.blob.scale] - frame.scale_eps;
        float scalemax = frame.scales[mode.blob.scale] + frame.scale_eps;
        // blobs = collect_blobs(frame.blob, 0, std::numeric_limits<float>::infinity());
        // blobs = collect_blobs(frame.blob, scalemin, scalemax);
        for(ScaleBlob *sb : mode.highlight.blobs){
        // for(ScaleBlob *sb : blobs){
          if(sb->scale < scalemin || sb->scale > scalemax)continue;
          for(ScaleBlob *succ : sb->succ){
            geometry.lines.push_back(sb->position);
            geometry.lines.push_back(succ->position);
            geometry.lines_c.push_back(sf::Color(255,255,255,120));
            geometry.lines_c.push_back(sf::Color(255,255,255,120));
          }
          for(ScaleBlob *pred : sb->pred){
            geometry.lines.push_back(sb->position);
            geometry.lines.push_back(pred->position);
            geometry.lines_c.push_back(sf::Color(80,80,80,80));
            geometry.lines_c.push_back(sf::Color(80,80,80,80));
          }
        }
      }

      // draw trajectory of highlighted blobs
      if(!mode.highlight.blobs.empty()){
        // std::queue<ScaleBlob*> traverse;
        // for(int i=0;i<mode.highlight.blobs.size();++i){
        //   traverse.push(mode.highlight.blobs[i]);
        // }
        // while(!traverse.empty()){
        //   ScaleBlob* curr = traverse.front();
        //   traverse.pop();
        //   for(ScaleBlob *succ : curr->succ){
        //     traverse.push(succ);
        //     geometry.lines.push_back(curr->position);
        //     geometry.lines.push_back(succ->position);
        //     geometry.lines_c.push_back(sf::Color(255,255,255,120));
        //     geometry.lines_c.push_back(sf::Color(255,255,255,120));
        //   }
        // }

        // // draw all branches in white.
        // for(std::vector<ScaleBlob*> path : paths){
        //   for(int j=0;j<path.size()-1;j++){
        //     for(int k=0;k<path[j]->succ.size();k++){
        //       geometry.lines.push_back(path[j]->position);
        //       geometry.lines.push_back(path[j]->succ[k]->position);
        //       geometry.lines_c.push_back(sf::Color::White);
        //       geometry.lines_c.push_back(sf::Color::White);
        //     }
        //   }
        // }

        // draw the longest trajectories.
        for(std::vector<ScaleBlob*> path : paths){
          float len  = float(path.size());
          float step = 1.f/len;
          for(int j=0;j<path.size()-1;j++){
            geometry.lines.push_back(path[j]->position);
            geometry.lines.push_back(path[j+1]->position);
            int r0 = 0   + int(200.f * step * j    );
            int g0 = 155 + int(100.f * step * j    );
            int b0 = 255 - int(250.f * step * j    );
            int r1 = 0   + int(200.f * step * (j+1));
            int g1 = 155 + int(100.f * step * (j+1));
            int b1 = 255 - int(250.f * step * j    );
            geometry.lines_c.push_back(sf::Color(r0,g0,b0,255));
            geometry.lines_c.push_back(sf::Color(r1,g1,b1,255));
          }
        }
      }
    }
  }

  for(vec3 v : mode.highlight.lines){
    geometry.lines.push_back(v);
    geometry.lines_c.push_back(sf::Color::Yellow);
  }

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
    
    // draw the blobs in paths at the current timestep.
    // if(!strcmp(mode.geom, "paths")){
    //   for(std::vector<ScaleBlob*> path : mode.highlight.paths){
    //     if(mode.timestep < path.size()){
    //       blobs.push_back(path[mode.timestep]);
    //     }
    //   }
    // }

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