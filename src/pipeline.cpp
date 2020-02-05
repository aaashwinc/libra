#include "pipeline.h"
#include "util.h"
#include <iostream>
#include <glm/glm.hpp>
#include <stdio.h>
#include <string.h>
#include <set>
#include <queue>
#include <limits>
#include <queue>
#include <SFML/Graphics.hpp>

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "blobmath.h"
#include "estimator.h"

using namespace std::chrono; 
using namespace std;

ReprMode::ReprMode(const char *name){
  this->name = name;
  this->geom = "none";
  blob.scale = 0;
  timestep   = 0;
  highlight.blobs = std::vector<ScaleBlob*>();
  highlight.lines = std::vector<vec3>();
  highlight.paths = std::vector<std::vector<ScaleBlob*>>();
  highlight.timestep = -1;
  highlight.locus    = vec3(0,0,0);
  highlight.path_smooth_alpha = 0.9f;
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
    data.tgmm_blob = 0;
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
void ArPipeline::path_highlight(ReprMode *rm, vec3 p, vec3 ray, bool diagnose, bool add){
  if(!get(rm->timestep).complete){
    printf("timestep not yet complete. break.\n");
  }
  ray = normalize(ray)*3.f;
  std::vector<ScaleBlob*> found;
  ArFrameData frame = get(rm->timestep);
  BSPTree<ScaleBlob> *bsptree = &(frame.bspblobs);

  // shoot ray out until it hits a blob.
  // imagine this ray as actually being a "cone" projecting outward,
  // with higher precision near the start.
  for(int i=0;i<100 && found.size()==0;++i){
    p+=ray;
    bsptree->find_within_distance(found, p, i+1);
    if(found.size()>0){
      break;
    }
  }
  ScaleBlob *locus = found[0];
  rm->highlight.highlight_loci.clear();
  rm->highlight.highlight_loci.push_back(found[0]);
}

void ArPipeline::repr_highlight(ReprMode *rm, vec3 p, vec3 ray, bool drawpaths, bool add){
  if(get(rm->timestep).complete){
    // printf("highlight along (%.2f %.2f %.2f) + (%.2f %.2f %.2f)\n",p.x,p.y,p.z, ray.x,ray.y,ray.z);
    ray = normalize(ray)*3.f;
    std::vector<ScaleBlob*> found;
    ArFrameData frame = get(rm->timestep);
    BSPTree<ScaleBlob> *bsptree = &(frame.bspblobs);

    printf("number of blobs = %d\n", bsptree->n);

    // shoot ray out until it hits a blob.
    for(int i=0;i<100 && found.size()==0;++i){
      p+=ray;
      bsptree->find_within_distance(found, p, i+1);
      if(found.size()>0){
        // if(drawpaths){
        //   printf("highlight blob %p:\n", found[0]);
        //   found[0]->print();
        // }
        if(!add){
          rm->highlight.highlight_loci.clear();
        }
        rm->highlight.highlight_loci.push_back(found[0]);

        found = std::vector<ScaleBlob*>();
        // found = bsptree->as_vector();
        bsptree->find_within_distance(found, p, 625);
      }
    }

    if(!add)rm->highlight.blobs.clear();
    for(int i=0;i<found.size();i++)rm->highlight.blobs.push_back(found[i]);

    if(drawpaths){
      // printf("drawpaths.\n");
      if(add){
        rm->highlight.paths = paths;
      }else{
        rm->highlight.paths.clear();
        for(std::vector<ScaleBlob*> path : paths){
          // if(path[0] == found[0]){
          //   rm->highlight.paths.push_back(path);
          // }
          for(ScaleBlob *blob : path){
            // printf("blob=%p\n,", blob);
            if(glm::distance(vec3(blob->position), p) < 5){
              rm->highlight.paths.push_back(path);
              break;
            }
            // break;
          }
        }
      }
    }else{
      rm->highlight.paths.clear();
    }
    rm->highlight.locus = p;
    rm->highlight.timestep = rm->timestep;
    // rm->highlight.paths = longest_paths(rm->highlight.blobs);
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
  // scalemax = 10000;collect_blobs
  // printf("(");
  // printf("collect_blobs %.2f %.2f\n", scalemin, scalemax);
  std::vector<ScaleBlob*> blobs;
  if(blob->scale >= scalemin || blob->scale == 0 || scalemin == -1){
    if(blob->scale <= scalemax || blob->scale == 0 || scalemax == -1){
      // printf(".");
      blobs.push_back(blob);
    }
    // printf("  scale = %.2f\n", blob->scale);
    for(ScaleBlob *child : blob->children){
      std::vector<ScaleBlob*> childblobs = collect_blobs(child, scalemin, scalemax);
      blobs.insert(blobs.end(), childblobs.begin(), childblobs.end());
    }
  }
  // printf(")");
  return blobs;
}


static std::vector<ScaleBlob*> detected_blobs(ScaleBlob *blob){
  // printf("peak= %.2f\n", blob->peakvalue);
  std::vector<ScaleBlob*> blobs;
  // printf("parent = %p\n", blob->parent);
  if(blob->parent == 0 || blob->peakvalue > blob->parent->peakvalue){
    blobs.push_back(blob);
    // printf("  pushing\n");
  }else{
    // printf("  NO!\n");
  }
  for(ScaleBlob *child : blob->children){
    std::vector<ScaleBlob*> childblobs = detected_blobs(child);
    blobs.insert(blobs.end(), childblobs.begin(), childblobs.end());
  }
  // printf(")");
  return blobs;
}

/* find all paths in the part of the dataset that has been
 * processed so far, so long as the path is greater than
 * minlen in length.
 *
 */

static bool fun_sort_blob_by_n(ScaleBlob* a, ScaleBlob* b){
  return a->n > b->n;
}
void ArPipeline::findpaths(int minlen, int maxframe, const char* mode){
  if(maxframe <= 0)maxframe = frames.size();
  if(maxframe > frames.size())maxframe = frames.size();
  std::vector<ScaleBlob*> allblobs;
  for(int i=0;i<maxframe;i++){   // for each frame
    if(frames[i].complete){           // if it is processed
                                      // collect blobs
      std::vector<ScaleBlob*> blobsi = collect_blobs(frames[i].blob, 0, std::numeric_limits<float>::infinity());
      if(!strcmp(mode, "tgmm")){
        // printf("tgmm!");
        blobsi = collect_blobs(frames[i].tgmm_blob, -1, -1);
      }
      std::sort(blobsi.begin(), blobsi.end(), fun_sort_blob_by_n);
      // for(auto sb : blobsi){
      //   printf("%.2f; ", sb->n);
      // }printf("\n");
      allblobs.insert(allblobs.end(), blobsi.begin(), blobsi.end());
      // minlen = int(i*0.75);
    }
  }
  printf("find paths > %d.\n", minlen);
  printf("find paths in %lu blobs.\n", allblobs.size());
  std::vector<std::vector<ScaleBlob*>> allpaths;
  if(!strcmp(mode, "quick"))   allpaths = longest_paths2(allblobs, minlen);
  if(!strcmp(mode, "tgmm"))    allpaths = longest_paths2(allblobs, minlen);
  if(!strcmp(mode, "longest")) allpaths = longest_paths2(allblobs, minlen);
  paths.clear();
  for(int i=0;i<allpaths.size();++i){
    if(allpaths[i].size() > minlen){
      paths.push_back(allpaths[i]);
    }
  }
  paths = allpaths;
  printf("found %lu paths.\n", paths.size());
}

// attempt to parallelize it. doesn't work though.
    // static void process_frame(ArExperiment *exp, ArFrameData *frame, int framen){
    //   ArFilter filter;
    //   filter.init(exp->get(framen));
    //   // filter.capture(exp->get(framen));
    //   printf("process %d\n", framen);
    //   ScaleBlob *blob             = filter.compute_blob_tree();


    //   std::vector<float> scales   = collect_scales(blob);
    //   frame->blob      = blob;
    //   frame->scales    = scales;
    //   frame->scale_eps = compute_epsilon(scales);
    //   frame->complete  = true;
    //   frame->bspblobs  = filter.get_bsp(10);

    //   BSPTree<ScaleBlob> *bsptree = &(frame->bspblobs);
    //   std::vector<ScaleBlob*> allblobs = collect_blobs(blob, 0, std::numeric_limits<float>::infinity());
      
    //   for(ScaleBlob *sb : allblobs){
    //     bsptree->insert(sb, sb->position);
    //   }
    //   filter.destroy();
    // }

// void ArPipeline::link_frames(int low, int high){
//   if(high > exp->high){
//     high = exp->high;
//     printf("pipeline.link_frames, low=%d, high=%d\n", low, high);
//   }
//   for(int frame=low;frame<=high-1;frame++){
//     if(!get(frame).complete)break;
    
//     BSPTree<ScaleBlob> *t0 = &frames[ frame - exp->low    ].bspblobs;
//     BSPTree<ScaleBlob> *t1 = &frames[ frame - exp->low +1 ].bspblobs;

//     std::vector<ScaleBlob*> v0 = t0->as_vector();
//     for(ScaleBlob *sb : v0){
//       std::vector<ScaleBlob*> potential;
//       t1->find_within_distance(potential, sb->position, 1000.f);
//       for(int i=0;i<potential.size();i++){

//         //  *** NOTE ***
//         // 1.0 means touching. I don't know what 2.0 means. This is somewhat
//         // arbitrary, but we notice that in the data, sometimes successor
//         // blobs aren't touching predecessor blobs, so we need a small margin.
//         // In any case, we next want to choose the potential successor with
//         // the smallest distance and closest scale.
//         //  *** **** ***

//         if(sb->n>1 && potential[i]->n>1 && sb->distance(potential[i]) <= 1.f){
//           if(sb->detCov/potential[i]->detCov < 2.f && potential[i]->detCov/sb->detCov < 8.f){
//             // volume cannot more than double or half.
//             sb->succ.push_back(potential[i]);
//             potential[i]->pred.push_back(sb);
//           }
//         }
//         // printf("%d.", i);
//       }
//       // printf("o");
//       ++itr;
//       // for(ScaleBlob *sb0 : sb->succ){
//       //   sb0->pred.push_back(sb);
//       // }
//     }
//   }
// }
/* store.buf[0] contains the output of the filter chain.
 * store.buf[1] contains the output of repr().
 */

void ArPipeline::link(int low, int high){
  printf("linking...");
  for(int frame = this->low(); frame <= high; ++frame){
    if(get(frame).complete && get(frame+1).complete);
    else continue;
    BSPTree<ScaleBlob> *t0 = &frames[ frame - exp->low    ].bspblobs;
    BSPTree<ScaleBlob> *t1 = &frames[ frame - exp->low +1 ].bspblobs;

    std::vector<ScaleBlob*> v0 = t0->as_vector();
    std::vector<ScaleBlob*> v1 = t1->as_vector();
    // clear predecessors so we can rewrite them.
    for(ScaleBlob *sb : v1){
      sb->pred.clear();
    }
    for(ScaleBlob *sb : v0){
      sb->succ.clear();       // clear successors so we can rewrite them.
      std::vector<ScaleBlob*> potential;
      ScaleBlob* closest = t1->find_closest(sb->position);
      potential.push_back(closest);
      t1->find_within_distance(potential, sb->position, 1000.f);
      for(int i=0;i<potential.size();i++){
        if(sb->n>1 && potential[i]->n>1 && sb->distance(potential[i]) <= 1.f){
          if(sb->detCov/potential[i]->detCov < 8.f && potential[i]->detCov/sb->detCov < 8.f){            // volume cannot more than double or half.
            sb->succ.push_back(potential[i]);
            potential[i]->pred.push_back(sb);
          }
        }
      }
      std::sort(sb->succ.begin(), sb->succ.end(), fun_sort_blob_by_n);
    }
  }
  printf("done.\n");

}
void ArPipeline::process(int low, int high){
  if(high > exp->high){
    high = exp->high;
    printf("pipeline.process, low=%d, high=%d\n", low, high);
  }
  for(int frame=low;frame<=high;frame++){
    printf("pipeline.process %d\n",frame);
    tick("");

    // THIS CAN BE PARALLELIZED {

        // ARFilter filter;
        filter.capture(exp->get(frame));
        // printf("process %d\n", frame)
        ScaleBlob *blob             = filter.compute_blob_tree();


        std::vector<float> scales   = collect_scales(blob);
        frames[frame - exp->low].blob      = blob;
        frames[frame - exp->low].scales    = scales;
        frames[frame - exp->low].scale_eps = compute_epsilon(scales);
        frames[frame - exp->low].complete  = true;
        frames[frame - exp->low].bspblobs  = filter.get_bsp(10);

        BSPTree<ScaleBlob> *bsptree = &frames[frame - exp->low].bspblobs;
        std::vector<ScaleBlob*> allblobs = collect_blobs(blob, 0, std::numeric_limits<float>::infinity());
        
        for(ScaleBlob *sb : allblobs){
          bsptree->insert(sb, sb->position);
        }

    // } THIS CAN BE PARALLELIZED

    // link with previous frame.
    printf("linking.\n");
    if(frame-1 >= this->low() && get(frame-1).complete){
      BSPTree<ScaleBlob> *t0 = &frames[ frame - exp->low -1 ].bspblobs;
      BSPTree<ScaleBlob> *t1 = &frames[ frame - exp->low    ].bspblobs;

      std::vector<ScaleBlob*> v0 = t0->as_vector();
      int itr = 0;
      for(ScaleBlob *sb : v0){
        std::vector<ScaleBlob*> potential;
        t1->find_within_distance(potential, sb->position, 1000.f);
        for(int i=0;i<potential.size();i++){

          //  *** NOTE ***
          // 1.0 means touching. I don't know what 2.0 means. This is somewhat
          // arbitrary, but we notice that in the data, sometimes successor
          // blobs aren't touching predecessor blobs, so we need a small margin.
          // In any case, we next want to choose the potential successor with
          // the smallest distance and closest scale.
          //  *** **** ***

          if(sb->n>1 && potential[i]->n>1 && sb->distance(potential[i]) <= 1.f){
            if(sb->n/potential[i]->n < 8.f && potential[i]->n/sb->n < 8.f){
              // volume cannot more than double or half.
              sb->succ.push_back(potential[i]);
              potential[i]->pred.push_back(sb);
            }
          }
          // printf("%d.", i);
        }
        std::sort(sb->succ.begin(), sb->succ.end(), fun_sort_blob_by_n);

        // for(ScaleBlob *next : sb->succ){
        //   printf("%.2f; ", next->n);
        // }printf("\n");
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
  if(!strcmp(mode.geom, "flow")){

    // blobs = collect_blobs(frame.blob, 0, std::numeric_limits<float>::infinity());
    // int i=0;
    // for(int x=0;x<)
    // for(ScaleBlob *sb : blobs){
    //   geometry.lines.push_back(sb.position)
    // }
  }
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


      if(!strcmp(mode.geom, "graph") || !strcmp(mode.geom, "succs")){
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


      // draw reference x- y- z- axes 
      geometry.lines.push_back(dvec3(0,0,0));
      geometry.lines_c.push_back(sf::Color(255,0,0,255));
      geometry.lines.push_back(dvec3(100,0,0));
      geometry.lines_c.push_back(sf::Color(255,0,0,255));
      geometry.lines.push_back(dvec3(0,0,0));
      geometry.lines_c.push_back(sf::Color(0,255,0,255));
      geometry.lines.push_back(dvec3(0,100,0));
      geometry.lines_c.push_back(sf::Color(0,255,0,255));
      geometry.lines.push_back(dvec3(0,0,0));
      geometry.lines_c.push_back(sf::Color(0,0,255,255));
      geometry.lines.push_back(dvec3(0,0,100));
      geometry.lines_c.push_back(sf::Color(0,0,255,255));
      
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
        // double path_smooth_beta = 1.f - mode.highlight.path_smooth_alpha;
        // printf("Draw %d paths.\n", mode.highlight.paths.size());

        for(std::vector<ScaleBlob*> path : mode.highlight.paths){

          std::vector<glm::dvec3> smoothed(path.size());

          if(true){   // smooth path
            int smooth_path = 5;
            for(int i=0;i<path.size();i++){
              int k = smooth_path;
              if(i-k < 0){
                k = i;
              }
              if(i+k >= path.size()){
                k = path.size() - i - 1;
              }
              int j0 = i-k;
              int j1 = i+k;
              float n=k*2+1;
              for(int j=j0;j<=j1;++j){
                smoothed[i] += path[j]->position;
              }
              smoothed[i] /= n;
              // smoothed[i] = path[i]->position;
            }
          }

          if(false){  // draw vectors
            smoothed = std::vector<glm::dvec3>(2);
            smoothed[0] = path[0]->position;
            smoothed[1] = path[path.size()-1]->position;
            // smoothed[1] = path[path.size()-1]->position - path[1]->position;
            // smoothed[1] *= 20.f/glm::length(smoothed[1]);
            // smoothed[1] += smoothed[0];
          }

          float len  = float(smoothed.size());
          // if(len<20)continue;
          float step = 1.f/len;
          glm::dvec3 weightedp = path[0]->position;
          for(int j=0;j<smoothed.size()-1;j++){
            // geometry.lines.push_back(path[j]->position);
            // geometry.lines.push_back(path[j+1]->position);
            // geometry.lines.push_back(weightedp);
            // glm::dvec3 weightedq = path[j+1]->position;
            
            // weightedp = (weightedp * mode.highlight.path_smooth_alpha) + (weightedq * path_smooth_beta);

            // geometry.lines.push_back(weightedp);

            geometry.lines.push_back(smoothed[j]);
            geometry.lines.push_back(smoothed[j+1]);

            const int SMOOTH   = 0;
            const int GRADIENT = 1;
            int pathcolormode = GRADIENT;

            if(pathcolormode == SMOOTH){
              int r0 = 0   + int(200.f * step * j    );
              int g0 = 155 + int(100.f * step * j    );
              int b0 = 255 - int(250.f * step * j    );
              int r1 = 0   + int(200.f * step * (j+1));
              int g1 = 155 + int(100.f * step * (j+1));
              int b1 = 255 - int(250.f * step * j    );
              geometry.lines_c.push_back(sf::Color(r0,g0,b0,255));
              geometry.lines_c.push_back(sf::Color(r1,g1,b1,255));
            }
            else if(pathcolormode == GRADIENT){
              glm::dvec3 direction = path[path.size()-1]->position - path[0]->position;
              float len = glm::length(direction);
              direction = glm::normalize(direction);
              int alpha = len*15;
              if(alpha>255)alpha = 255;
              int r,g,b;
              r = 127 + direction.x*127;
              g = 127 + direction.y*127;
              b = 127 + direction.z*127;
              geometry.lines_c.push_back(sf::Color(r,g,b,alpha));
              geometry.lines_c.push_back(sf::Color(r,g,b,alpha));
            }
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
          // if(sb->scale < scalemin || sb->scale > scalemax)continue;
          for(ScaleBlob *succ : sb->succ){
            geometry.lines.push_back(sb->position);
            geometry.lines.push_back(succ->position);
            geometry.lines_c.push_back(sf::Color(255,255,255,200));
            geometry.lines_c.push_back(sf::Color(255,255,255,200));
          }
          for(ScaleBlob *pred : sb->pred){
            geometry.lines.push_back(sb->position);
            geometry.lines.push_back(pred->position);
            geometry.lines_c.push_back(sf::Color(0,40,0,200));
            geometry.lines_c.push_back(sf::Color(0,40,0,200));
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
/**
 * Chooses how to represent the volume-rendered data.
 * Eg. we may represent the data as:
 *   - "plain"     : raw data
 *   - "blobs"     : depicting the blobs recovered from the data
 *   - "gaussian"  : rendering a blurred version of the data at some sigma
 *   - "laplacian" : laplacian of gaussian filter at some sigma
*/
Nrrd *ArPipeline::repr(ReprMode &mode, bool force){
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
    // filter.clear();
    // std::vector<ScaleBlob*> blobs;
    // ScaleBlob *a = new ScaleBlob();
    // a->pass(vec3(10,10,10), 10);
    // a->pass(vec3(15,15,15), 30);
    // a->pass(vec3(10,20,20), 10);
    // a->commit();
    // blobs.push_back(a);
    // filter.draw_blobs(blobs, false);
    // filter.max1();
    // filter.normalize(3.f);
    // filter.commit(store.buf[1]);
    ArFrameData frame = get(mode.timestep);
    // float scale = 4.f;
    filter.clear();
    // filter.capture(exp->get(mode.timestep));
    ScaleBlob* blob = frame.blob->children[0]->children[0]->children[0];
    std::vector<ScaleBlob*> highlight;
    highlight.push_back(blob);
    filter.draw_blobs(highlight);
    // ivec3 points[]={ivec3(100, 129, 194), ivec3(36, 103, 193), ivec3(60, 145, 185), ivec3(142, 35, 185), ivec3(133, 1, 198), ivec3(45, 66, 190), ivec3(82, 81, 196), ivec3(145, 93, 194), ivec3(71, 50, 161), ivec3(113, 79, 197), ivec3(143, 25, 194), ivec3(87, 44, 192), ivec3(82, 107, 194), ivec3(6, 142, 185), ivec3(29, 128, 195), ivec3(72, 139, 189), ivec3(101, 76, 196), ivec3(144, 59, 153), ivec3(103, 15, 182), ivec3(81, 137, 193), ivec3(71, 142, 196), ivec3(122, 90, 168), ivec3(118, 92, 184), ivec3(113, 32, 197), ivec3(66, 116, 198), ivec3(121, 93, 192), ivec3(48, 87, 187), ivec3(2, 86, 175), ivec3(110, 4, 140), ivec3(65, 65, 180), ivec3(27, 43, 196), ivec3(133, 94, 137), ivec3(134, 95, 128), ivec3(147, 83, 164), ivec3(92, 120, 119), ivec3(132, 64, 181), ivec3(143, 14, 127), ivec3(138, 22, 138), ivec3(20, 102, 117), ivec3(55, 5, 197), ivec3(29, 79, 197), ivec3(125, 71, 137), ivec3(83, 148, 109), ivec3(145, 3, 190), ivec3(12, 80, 186), ivec3(27, 148, 128), ivec3(41, 145, 187), ivec3(102, 83, 150), ivec3(101, 108, 141), ivec3(109, 144, 117), ivec3(145, 7, 115), ivec3(4, 89, 162), ivec3(20, 131, 69), ivec3(14, 126, 131), ivec3(83, 122, 153), ivec3(124, 121, 157), ivec3(137, 58, 81), ivec3(138, 25, 3), ivec3(93, 86, 15), ivec3(49, 132, 80), ivec3(86, 148, 3), ivec3(59, 114, 165), ivec3(79, 125, 129), ivec3(89, 113, 126), ivec3(9, 54, 167), ivec3(5, 72, 132), ivec3(44, 99, 172), ivec3(30, 97, 92), ivec3(39, 134, 27), ivec3(86, 83, 182), ivec3(88, 93, 172), ivec3(81, 118, 188), ivec3(89, 101, 161), ivec3(92, 47, 53), ivec3(114, 141, 87), ivec3(132, 144, 77), ivec3(79, 84, 38), ivec3(10, 60, 76), ivec3(23, 101, 136), ivec3(1, 86, 55), ivec3(72, 15, 154), ivec3(143, 148, 169), ivec3(144, 53, 51), ivec3(144, 136, 15), ivec3(143, 102, 73), ivec3(130, 110, 5), ivec3(130, 108, 13), ivec3(72, 67, 127), ivec3(47, 93, 122), ivec3(135, 6, 4), ivec3(26, 3, 176), ivec3(109, 106, 130), ivec3(121, 144, 128), ivec3(147, 50, 144), ivec3(131, 143, 22), ivec3(90, 120, 26), ivec3(13, 147, 97), ivec3(7, 43, 141), ivec3(50, 46, 149), ivec3(20, 15, 80), ivec3(102, 104, 150), ivec3(115, 121, 63), ivec3(148, 92, 51), ivec3(92, 134, 3), ivec3(59, 84, 151), ivec3(47, 147, 88), ivec3(38, 130, 93), ivec3(30, 148, 45), ivec3(111, 62, 80), ivec3(21, 35, 157), ivec3(69, 10, 142), ivec3(94, 45, 46), ivec3(139, 116, 44), ivec3(117, 122, 56), ivec3(144, 53, 44), ivec3(125, 56, 51), ivec3(82, 121, 58), ivec3(86, 86, 104), ivec3(43, 62, 3), ivec3(132, 3, 16), ivec3(109, 13, 28), ivec3(80, 18, 61), ivec3(88, 54, 11), ivec3(60, 35, 3), ivec3(84, 11, 104), ivec3(71, 46, 45), ivec3(3, 51, 91), ivec3(55, 77, 79), ivec3(56, 78, 85), ivec3(71, 59, 66), ivec3(74, 48, 32), ivec3(43, 63, 148), ivec3(68, 9, 103), ivec3(56, 38, 79), ivec3(33, 2, 70), ivec3(70, 58, 78), ivec3(92, 87, 4), ivec3(54, 111, 87), ivec3(56, 50, 93), ivec3(45, 18, 41), ivec3(78, 71, 11), ivec3(59, 35, 112), ivec3(108, 77, 105), ivec3(142, 59, 3), ivec3(7, 125, 105), ivec3(135, 147, 160), ivec3(82, 127, 137), ivec3(86, 99, 85), ivec3(40, 122, 123), ivec3(29, 148, 56), ivec3(40, 135, 37), ivec3(145, 134, 6), ivec3(25, 50, 59), ivec3(86, 43, 107), ivec3(126, 65, 115), ivec3(40, 28, 146), ivec3(105, 17, 40), ivec3(102, 115, 33), ivec3(26, 128, 184), ivec3(102, 148, 158), ivec3(122, 123, 88), ivec3(86, 121, 38), ivec3(36, 91, 80), ivec3(116, 102, 76), ivec3(140, 25, 25), ivec3(145, 136, 63), ivec3(48, 98, 46), ivec3(115, 35, 24), ivec3(127, 33, 75), ivec3(59, 38, 68), ivec3(108, 71, 71), ivec3(145, 126, 180), ivec3(142, 147, 195), ivec3(6, 125, 167), ivec3(26, 99, 174), ivec3(96, 12, 152), ivec3(24, 63, 199), ivec3(131, 76, 152), ivec3(78, 146, 162), ivec3(78, 109, 152), ivec3(85, 121, 48), ivec3(56, 110, 99), ivec3(17, 115, 24), ivec3(128, 28, 140), ivec3(52, 23, 137), ivec3(147, 101, 95), ivec3(144, 0, 49), ivec3(148, 147, 71), ivec3(121, 111, 196), ivec3(56, 74, 176), ivec3(86, 85, 189), ivec3(49, 138, 196), ivec3(33, 67, 177), ivec3(126, 56, 190), ivec3(147, 9, 152), ivec3(5, 113, 188), ivec3(62, 119, 146), ivec3(139, 58, 63), ivec3(123, 131, 172), ivec3(111, 67, 95), ivec3(105, 145, 24), ivec3(140, 143, 29), ivec3(95, 113, 1), ivec3(22, 57, 129), ivec3(45, 32, 161), ivec3(92, 46, 59), ivec3(144, 138, 21), ivec3(74, 25, 101), ivec3(53, 74, 1), ivec3(140, 117, 121), ivec3(102, 131, 101), ivec3(3, 57, 102), ivec3(55, 84, 46), ivec3(21, 148, 139), ivec3(1, 76, 156), ivec3(107, 101, 90), ivec3(24, 141, 176), ivec3(94, 70, 167), ivec3(130, 72, 195), ivec3(68, 90, 180), ivec3(0, 9, 195), ivec3(144, 79, 109), ivec3(62, 27, 196), ivec3(98, 139, 183), ivec3(2, 136, 141), ivec3(16, 117, 33), ivec3(131, 117, 143), ivec3(1, 45, 181), ivec3(131, 34, 128), ivec3(106, 102, 68), ivec3(48, 146, 67), ivec3(60, 79, 139), ivec3(64, 138, 44), ivec3(98, 145, 70), ivec3(125, 14, 48), ivec3(55, 79, 71), ivec3(134, 77, 47), ivec3(33, 81, 91), ivec3(107, 50, 197), ivec3(78, 25, 196), ivec3(148, 119, 190), ivec3(34, 27, 189), ivec3(137, 0, 177), ivec3(6, 132, 125), ivec3(92, 146, 82), ivec3(81, 124, 69), ivec3(53, 130, 65), ivec3(5, 100, 140), ivec3(6, 130, 118), ivec3(133, 144, 10), ivec3(130, 98, 161), ivec3(90, 118, 12), ivec3(57, 37, 11), ivec3(77, 27, 104), ivec3(132, 142, 16), ivec3(21, 73, 134), ivec3(135, 25, 36), ivec3(102, 84, 73), ivec3(72, 8, 28), ivec3(144, 0, 57), ivec3(139, 54, 97), ivec3(145, 139, 118), ivec3(31, 80, 120), ivec3(94, 51, 3), ivec3(1, 57, 198), ivec3(137, 83, 194), ivec3(147, 78, 150), ivec3(76, 148, 139), ivec3(90, 3, 179), ivec3(98, 4, 166), ivec3(58, 146, 122), ivec3(13, 123, 120), ivec3(73, 118, 114), ivec3(103, 117, 102), ivec3(35, 102, 159), ivec3(124, 20, 84), ivec3(7, 102, 149), ivec3(100, 73, 178), ivec3(53, 5, 143), ivec3(77, 23, 120), ivec3(144, 101, 84), ivec3(144, 121, 67), ivec3(134, 94, 58), ivec3(125, 54, 41), ivec3(146, 30, 96), ivec3(35, 90, 101), ivec3(142, 95, 13), ivec3(26, 127, 78), ivec3(71, 12, 95), ivec3(84, 140, 99), ivec3(42, 39, 129), ivec3(24, 94, 128), ivec3(139, 56, 88), ivec3(132, 95, 67), ivec3(126, 48, 170), ivec3(100, 33, 16), ivec3(85, 148, 117), ivec3(146, 127, 171), ivec3(108, 74, 60), ivec3(50, 102, 28), ivec3(68, 37, 187), ivec3(147, 0, 20), ivec3(104, 35, 4), ivec3(19, 113, 16), ivec3(123, 1, 123), ivec3(147, 22, 49), ivec3(60, 144, 4), ivec3(145, 24, 39), ivec3(12, 146, 88), ivec3(30, 14, 196), ivec3(55, 123, 198), ivec3(136, 83, 94), ivec3(85, 68, 150), ivec3(104, 88, 135), ivec3(21, 70, 165), ivec3(111, 57, 75), ivec3(11, 59, 67), ivec3(87, 7, 84), ivec3(68, 2, 123), ivec3(107, 147, 16), ivec3(69, 68, 34), ivec3(77, 3, 58), ivec3(85, 67, 1), ivec3(66, 33, 87), ivec3(21, 16, 90), ivec3(111, 101, 80), ivec3(3, 55, 175), ivec3(10, 56, 55), ivec3(31, 1, 134), ivec3(96, 22, 192), ivec3(98, 96, 118), ivec3(123, 34, 90), ivec3(146, 130, 192), ivec3(83, 35, 146), ivec3(1, 91, 196), ivec3(8, 112, 197), ivec3(45, 122, 104), ivec3(50, 147, 83), ivec3(136, 108, 196), ivec3(128, 17, 60), ivec3(146, 116, 109), ivec3(60, 82, 118), ivec3(1, 54, 113), ivec3(85, 100, 75), ivec3(84, 42, 98), ivec3(40, 62, 13), ivec3(41, 138, 49), ivec3(22, 99, 133), ivec3(145, 138, 110), ivec3(57, 94, 161), ivec3(56, 112, 156), ivec3(148, 43, 198), ivec3(104, 133, 150), ivec3(120, 13, 96), ivec3(137, 60, 69), ivec3(94, 40, 125), ivec3(148, 118, 139), ivec3(130, 117, 136), ivec3(65, 134, 31), ivec3(132, 144, 84), ivec3(52, 134, 11), ivec3(103, 1, 105), ivec3(94, 44, 36), ivec3(70, 33, 77), ivec3(44, 16, 52), ivec3(141, 28, 8), ivec3(24, 77, 159), ivec3(52, 52, 113), ivec3(92, 35, 54), ivec3(4, 125, 157), ivec3(115, 36, 174), ivec3(82, 121, 148), ivec3(54, 81, 105), ivec3(80, 126, 79), ivec3(137, 18, 172), ivec3(82, 122, 89), ivec3(16, 128, 140), ivec3(22, 119, 147), ivec3(138, 60, 75), ivec3(127, 58, 13), ivec3(71, 11, 40), ivec3(61, 32, 101), ivec3(75, 26, 25), ivec3(113, 41, 184), ivec3(92, 122, 111), ivec3(114, 34, 14), ivec3(38, 72, 121), ivec3(127, 56, 20), ivec3(108, 12, 16), ivec3(51, 38, 172), ivec3(124, 4, 77), ivec3(31, 2, 123), ivec3(1, 91, 191), ivec3(49, 145, 74), ivec3(70, 67, 43), ivec3(11, 147, 198), ivec3(145, 124, 78), ivec3(30, 120, 171), ivec3(44, 24, 35), ivec3(20, 84, 37), ivec3(126, 92, 113), ivec3(104, 116, 112), ivec3(137, 96, 4), ivec3(5, 148, 104), ivec3(26, 38, 147), ivec3(136, 76, 34), ivec3(138, 68, 45), ivec3(139, 118, 33), ivec3(128, 112, 32), ivec3(73, 69, 27), ivec3(88, 71, 157), ivec3(103, 6, 71), ivec3(15, 39, 186), ivec3(15, 63, 154), ivec3(97, 84, 25), ivec3(1, 87, 48), ivec3(19, 86, 46), ivec3(78, 2, 50), ivec3(107, 15, 192), ivec3(21, 104, 126), ivec3(97, 140, 49), ivec3(20, 87, 32), ivec3(143, 0, 46), ivec3(120, 121, 49), ivec3(120, 18, 107), ivec3(141, 146, 186), ivec3(100, 118, 184), ivec3(148, 24, 66), ivec3(14, 37, 174), ivec3(30, 31, 177), ivec3(21, 131, 63), ivec3(46, 22, 154), ivec3(126, 65, 109), ivec3(147, 61, 111), ivec3(44, 119, 111), ivec3(148, 100, 170), ivec3(35, 147, 38), ivec3(64, 41, 54), ivec3(144, 90, 26), ivec3(127, 19, 70), ivec3(83, 23, 81), ivec3(52, 53, 46), ivec3(57, 84, 39), ivec3(141, 0, 40), ivec3(146, 72, 94), ivec3(146, 137, 103), ivec3(13, 144, 81), ivec3(39, 129, 86), ivec3(91, 142, 149), ivec3(94, 60, 109), ivec3(124, 53, 33), ivec3(103, 0, 100), ivec3(97, 0, 91), ivec3(32, 2, 116), ivec3(40, 63, 159), ivec3(68, 14, 48), ivec3(83, 118, 95), ivec3(25, 102, 142), ivec3(75, 29, 40), ivec3(126, 64, 128), ivec3(134, 146, 144), ivec3(98, 84, 33), ivec3(96, 140, 56), ivec3(116, 122, 70), ivec3(26, 1, 143), ivec3(47, 148, 95), ivec3(35, 148, 99), ivec3(87, 86, 96), ivec3(29, 111, 72), ivec3(50, 92, 56), ivec3(53, 79, 93), ivec3(88, 50, 23), ivec3(37, 63, 23), ivec3(105, 100, 98), ivec3(34, 136, 107), ivec3(147, 121, 91), ivec3(146, 136, 95), ivec3(126, 2, 26), ivec3(14, 148, 152), ivec3(7, 90, 147), ivec3(123, 101, 91), ivec3(117, 66, 199), ivec3(120, 140, 188), ivec3(41, 137, 58), ivec3(98, 18, 47), ivec3(105, 8, 52), ivec3(145, 145, 52), ivec3(61, 69, 199), ivec3(13, 143, 75), ivec3(114, 11, 126), ivec3(42, 146, 16), ivec3(19, 147, 8), ivec3(96, 136, 13), ivec3(75, 15, 86), ivec3(133, 35, 107), ivec3(99, 84, 42), ivec3(78, 84, 45), ivec3(97, 64, 119), ivec3(17, 129, 74), ivec3(87, 19, 141), ivec3(120, 123, 78), ivec3(3, 148, 109), ivec3(105, 8, 66), ivec3(132, 79, 55), ivec3(144, 145, 39), ivec3(52, 135, 22), ivec3(40, 147, 26), ivec3(112, 65, 184), ivec3(31, 17, 166), ivec3(104, 145, 35), ivec3(100, 96, 125), ivec3(34, 143, 114), ivec3(56, 51, 100), ivec3(129, 148, 60), ivec3(61, 4, 133), ivec3(102, 114, 27), ivec3(147, 22, 57), ivec3(102, 84, 84), ivec3(141, 147, 176), ivec3(131, 2, 155), ivec3(119, 55, 148), ivec3(108, 75, 48), ivec3(128, 18, 67), ivec3(106, 67, 138), ivec3(51, 98, 129), ivec3(126, 54, 27), ivec3(130, 147, 36)};
    // std::vector<ivec3> pointsv;
    // pointsv.assign(points, points+(sizeof(points)/sizeof(points[0])));
    // filter.highlight(pointsv);

    // DiscreteKernel kernel = filter.gaussian(scale, int(scale*4));
    // filter.set_kernel(kernel);
    // filter.max1();
    // filter.filter();
    filter.commit(store.buf[1]);
    // kernel.destroy();
    // return (last_nrrd = store.buf[1]);
    return store.buf[1];
  }
  if(!get(mode.timestep).complete){
    return exp->get(mode.timestep);
  }

  static ReprMode last_repr("");
  static Nrrd *last_nrrd;
  if(mode == last_repr && !force){
    // printf("repr %s unchanged.\n",mode.name);
    return last_nrrd;
  }
  last_repr  = mode;
  // printf("repr %s\n",mode.name);

  if(!strcmp(mode.name, "flow")){
    // visualize the flow of GFP throughout the experiment.
    
    printf("repr %s\n", mode.name); 
    ArFrameData frame = get(mode.timestep);   
    filter.capture(exp->get(exp->low));

    // use a gaussian kernel
    float scale = frame.scales[mode.blob.scale];
    DiscreteKernel kernel = filter.gaussian(scale, int(scale*4));
    filter.set_kernel(kernel);

    // perform laplacian of gaussian
    tick("flow");
    filter.max1();
    filter.filter();
    filter.laplacian3d();
    filter.normalize();

    // get maxima at t=0
    std::vector<std::vector<glm::vec3>> paths;

    std::vector<ivec3> maxima = filter.find_maxima();
    for(ivec3 maximum : maxima){
      std::vector<vec3> path;
      path.push_back(vec3(maximum.x, maximum.y, maximum.z));
      paths.push_back(path);
    }

    // highlight.push_back(filter.find_maxima());

    tick("t=0");

    for(int i=0;i<10;i++){
      printf("lap of gaus\n");
      // laplacian of gaussian of frame i
      filter.capture(exp->get(exp->low + i));
      filter.max1();          // max
      filter.filter();        // gaussian
      filter.laplacian3d();   // laplacian
      filter.normalize();     // normalize
      printf("hill climb\n");
      // hill-climb into the next frame.
      for(int p=0;p<paths.size();p++){
        ivec3 point(paths[p][i].x, paths[p][i].y, paths[p][i].z);
        ivec3 hill = filter.hill_climb(point);
        paths[p].push_back(hill);
      }
      // tick("hill-climb");
    }

    printf("found %lu paths.\n", paths.size());
    filter.clear();
    for(int i=0;i<paths.size();i++){
      for(int j=0;j<paths[i].size()-1;j++){
        // printf("  line from %.2f %.2f %.2f to %.2f %.2f %.2f\n",
         // paths[i][j].x,   paths[i][j].y,   paths[i][j].z,
         // paths[i][j+1].x, paths[i][j+1].y, paths[i][j+1].z);
         filter.rasterlineadd(paths[i][j], paths[i][j+1], 1.f, 1.f);
      }
    }
    printf("gaussian.\n");

    DiscreteKernel kernel2 = filter.gaussian(1.f, 5);
    filter.set_kernel(kernel2);
    filter.filter();        // gaussian
    filter.normalize();
    kernel2.destroy();

    kernel.destroy();

    // filter.clear();
    // filter.highlight(highlight[0]);
    // filter.highlight(highlight[1]);

    filter.commit(store.buf[1]);
    return (last_nrrd = store.buf[1]);
  }
  if(!strcmp(mode.name, "blobs_all")){
    
    ArFrameData frame = get(mode.timestep);
    float scale = frame.scales[mode.blob.scale];
    // scale = 2.f;
    // filter.capture(exp->get(mode.timestep));
    DiscreteKernel kernel = filter.gaussian(scale/1, int(scale*4));
    filter.set_kernel(kernel);
    filter.max1();
    filter.filter();
    filter.laplacian3d();
    std::vector<ScaleBlob*> blobs = filter.find_blobs();
    filter.clear();
    // filter.draw_blobs(blobs);
    // filter.commit(store.buf[1]);
    kernel.destroy();
    // return (last_nrrd = store.buf[1]);
    // float scalemin = frame.scales[mode.blob.scale] - frame.scale_eps;
    // float scalemax = frame.scales[mode.blob.scale] + frame.scale_eps;
    // printf("mode %s. view scale %d with %.2f %.2f\n", mode.name, scalemin, scalemax);
    // std::vector<ScaleBlob*> blobs = collect_blobs(frame.blob, scalemin, scalemax);

    std::vector<ScaleBlob*> only;
    Estimator estimator;
    // filter.clear();
    ScaleBlob estimated_blob;
    only.push_back(&estimated_blob);
    for(int i=1;i<blobs.size();i++){
      estimated_blob = estimator.fit(exp->get(mode.timestep), blobs[i]);
      filter.draw_blobs(only, "m+");
    }
    printf("done blobs_all.\n");
    // filter.difference_image(exp->get(mode.timestep));
    filter.commit(store.buf[2]);
    return (last_nrrd = store.buf[2]);   
  }
  if(!strcmp(mode.name, "tgmm")){
    // printf("mode = tgmm\n");
    ArFrameData frame = get(mode.timestep);
    // printf("size[%d] = %p = 0", mode.timestep, frame.tgmm_blob);
    std::vector<ScaleBlob*> blobs = frame.tgmm_blob->children;
    // std::vector<ScaleBlob*> blobs = collect_blobs(frame.tgmm_blob, -1, -1);
    printf("draw %p = %lu blobs.\n", frame.tgmm_blob, blobs.size());
    // filter.clear();
    filter.capture(exp->get(mode.timestep));
    filter.draw_blobs(blobs, ".m");
    filter.commit(store.buf[2]);
    return (last_nrrd = store.buf[2]);   
  }
  if(!strcmp(mode.name, "blobs") || !strcmp(mode.name, "blobs_succs")){
    ArFrameData frame = get(mode.timestep);
    float scalemin = frame.scales[mode.blob.scale] - frame.scale_eps;
    float scalemax = frame.scales[mode.blob.scale] + frame.scale_eps;
    // printf("mode %s. view scale %d with %.2f %.2f\n", mode.name, scalemin, scalemax);
    std::vector<ScaleBlob*> blobs = collect_blobs(frame.blob, scalemin, scalemax);
    // std::vector<ScaleBlob*> blobs = collect_blobs(frame.blob, -1, -1);
    
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
    // for(int i=0;i<mode.highlight.highlight_loci.size();i++){
    //   blobs.push_back(mode.highlight.highlight_loci[i]);
    // }
    // blobs.push_back(mode.highlight.highlight_loci);
    // blobs.insert(blobs.end(), mode.highlight.highlight_loci.begin(), mode.highlight.highlight_loci.end());
    filter.capture(exp->get(mode.timestep));
    filter.draw_blobs(blobs, ".m");

    // for(ScaleBlob *sb : mode.highlight.blobs){
    // if(mode.highlight.highlight_loci.size() > 0){
    //   // printf("highlight %lu; scale = %.2f \n", mode.highlight.highlight_loci.size(), mode.highlight.highlight_loci[0]->scale);
    //   // filter.color_blobs(mode.highlight.highlight_loci, 2.f);
    // }
    // if(mode.highlight.highlight_loci.size() > 1){
    //   int i=0;
    //   int j=0;
    //   for(int i=0;i<mode.highlight.highlight_loci.size();++i){
    //     for(int j=0;j<mode.highlight.highlight_loci.size();++j){
    //       printf("  distance %d %d -- %.4f\n", i, j, 
    //         mode.highlight.highlight_loci[i]
    //           ->distance(
    //         mode.highlight.highlight_loci[j])
    //         );
    //     }
    //   }
    // }
    // }

    // show all successors of  blob.
    // if(!strcmp(mode.name, "blobs_succs")){
    //   BSPTree<ScaleBlob> *t1 = &frames[ mode.highlight.timestep - exp->low    ].bspblobs;
    //   std::vector<ScaleBlob*> succs;
    //   for(ScaleBlob *sb : mode.highlight.blobs){
    //     std::vector<ScaleBlob*> potential;
    //     t1->find_within_distance(potential, sb->position, 1000.f);
    //     for(int i=0;i<potential.size();i++){
    //       if(sb->n>1 && potential[i]->n>1 && sb->distance(potential[i]) <= 1.f){
    //         // if(sb->detCov/potential[i]->detCov < 2.f && potential[i]->detCov/sb->detCov < 2.f){
    //           // volume cannot more than double or half.
    //           succs.push_back(potential[i]);
    //           potential[i]->pred.push_back(sb);
    //         // }
    //       }
    //       // printf("%d.", i);
    //     }
    //   }
    //   filter.color_blobs(succs, 4.f);
    // }
    filter.commit(store.buf[1]);
    return (last_nrrd = store.buf[1]);
  }
  if(!strcmp(mode.name, "gaussian")){
    // printf("mode %s.\n", mode.name);
    ArFrameData frame = get(mode.timestep);
    // float scale = frame.scales[mode.blob.scale];
    float scale = 2.f;
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
    scale = 2.f;
    filter.capture(exp->get(mode.timestep));

    // DiscreteKernel kernel1 = filter.gaussian(3.f, int(3.f*4));;
    // DiscreteKernel kernel2 = filter.gaussian(1.f, int(1.f*4));;

    DiscreteKernel kernel1 = filter.gaussian(3.f, int(3.f*4));;
    DiscreteKernel kernel2 = filter.gaussian(2.f, int(2.f*4));;

    filter.max1();    
    filter.set_kernel(kernel1);
    filter.filter();
    filter.laplacian3d();
    filter.normalize(0.5f);
    filter.set_kernel(kernel2);
    filter.filter();
    // filter.filter();
    // filter.scale(5.f);
    // filter.threshold(0.f,1.f);
    filter.commit(store.buf[1]);
    kernel1.destroy();
    kernel2.destroy();
    return (last_nrrd = store.buf[1]);
  }
  if(!strcmp(mode.name, "masked")){
    // printf("mode %s.\n", mode.name);
    ArFrameData frame = get(mode.timestep);
    float scale = frame.scales[mode.blob.scale];
    filter.capture(exp->get(mode.timestep));
    DiscreteKernel kernel = filter.gaussian(scale, int(scale*4));
    filter.set_kernel(kernel);
    filter.max1();
    filter.filter();
    // filter.laplacian3d();
    // filter.filter();
    filter.laplacianmasked();
    filter.filter();
    filter.normalize(1.f);
    // filter.scale(5.f);
    filter.threshold(0.f,1.f);
    filter.commit(store.buf[1]);
    kernel.destroy();
    return (last_nrrd = store.buf[1]);
  }
}
/*
 * emit the current representation
 * as a set of .nrrd files.         ASSUMES 3 DIMENSIONS
 */
void ArPipeline::emit(ReprMode &mode, std::string suffix, int low, int high){
  // printf("emit %d %d\n", low, high);
  for(int i=low; i<=high; i++){
    if(i < exp->low)continue;
    if(i > exp->high)continue;
    mode.timestep = i;
    printf("emit to %s\n", (exp->getfilepath(i)+suffix).c_str());
    Nrrd *nrrdfloat = repr(mode);
    Nrrd *nrrdshort = nrrdNew();
    nrrdConvert(nrrdshort, nrrdfloat, nrrdTypeUShort);
    int size = nrrdshort->axis[0].size * nrrdshort->axis[1].size * nrrdshort->axis[2].size;
    short *datashort = (short*)nrrdshort->data;
    float *datafloat = (float*)nrrdfloat->data;
    for(int i=0;i<size;++i){
      datashort[i] = (unsigned short) (datafloat[i] * 32767.0);
    }
    // nrrdCopy(out, asfloat);
    nrrdSave((exp->getfilepath(i)+suffix).c_str(), nrrdshort, NULL);
    nrrdNuke(nrrdshort);
  }
}


  /* Save/Load processed pipeline. format:
   * filename as [filepath_0].pipeline
   * [int] number of frames processed sequentially from 0.
   * <list of blobs:
   *    [int] address of blob
   *    ... data ...
   * 
   * >
   */

void ArPipeline::save(){

#define WRITET(T, x) {T xx = (T)x; fwrite(&(xx), sizeof(T), 1, file);}
#define WRITE(x)    {fwrite(&(x), sizeof(x), 1, file);}

  std::string path0 = exp->getfilepath(exp->low);
  std::replace(path0.begin(), path0.end(), '/', '-');
  path0 = "../rsc/store/s" + path0;

  printf("writing to : %s\n", path0.c_str());

  FILE *file = fopen((path0 + ".pipeline").c_str(),"wb");

  // fwrite("hello!",sizeof(char),6,file);

  // count number of frames that are processed.
  int nframes = 0;
  for(int i=0;i<frames.size();i++){
    if(!frames[i].complete){
      break;
    }else{
      nframes = i+1;
    }
  }

  WRITET(int, nframes);

  for(int i=0;i<nframes;++i){
    // fwrite("ff", sizeof(char), 2, file);
    // write address of root scaleblob.
    WRITET(ScaleBlob*, frames[i].blob);
    // fwrite(&frames[i].blob, sizeof(ScaleBlob*), 1, file);
    std::vector<ScaleBlob*> allblobs = frames[i].bspblobs.as_vector();
    
    // write total number of blobs in this frame.
    WRITET(int, allblobs.size());
    // fwrite(&nblobs, sizeof(int), 1, file);

    for(ScaleBlob *sb : allblobs){
      // write each blob
      WRITET(ScaleBlob*, sb);

      WRITET(int, sb->children.size());
      for(ScaleBlob *s : sb->children){
        WRITE(s);
      }

      WRITET(int, sb->pred.size());
      for(ScaleBlob *s : sb->pred){
        WRITE(s);
      }

      WRITET(int, sb->succ.size());
      for(ScaleBlob *s : sb->succ){
        WRITE(s);
      }
      WRITE(sb->parent);
      WRITE(sb->mode);
      WRITE(sb->position);
      WRITE(sb->shape);
      WRITE(sb->fshape);
      WRITE(sb->timestep);
      WRITE(sb->covariance);
      WRITE(sb->invCov);
      WRITE(sb->detCov);
      WRITE(sb->pdfCoef);
      WRITE(sb->min);
      WRITE(sb->max);
      WRITE(sb->scale);
      WRITE(sb->n);
      WRITE(sb->npass);
      WRITE(sb->peakvalue);
    }
  }

  fclose(file);

  file = fopen((path0 + ".paths").c_str(),"wb");
  printf("write %lu\n", paths.size());
  WRITET(int, paths.size());
  for(std::vector<ScaleBlob*> path : paths){
    WRITET(int, path.size());
    for(ScaleBlob* s : path){
      WRITE(s);
    }
  }

  fclose(file);

  file = fopen((path0 + ".paths.txt").c_str(), "w");
  for(std::vector<ScaleBlob*> path : paths){
    for(ScaleBlob* blob : path){
      fprintf(file, "%.2f %.2f %.2f; ", blob->position.x, blob->position.y, blob->position.z);
    }
    fprintf(file, "\n");
  }

}
#undef WRITE
#undef WRITET
void ArPipeline::load(){
//   printf("loading...\n");
  std::string path0 = exp->getfilepath(exp->low);
  std::replace(path0.begin(), path0.end(), '/', '-');
  path0 = "../rsc/store/s" + path0;

  std::string path0pipeline = path0 + ".pipeline";
  std::string path0paths    = path0 + ".paths";

  if(access(path0pipeline.c_str(), F_OK) == -1)return;
  FILE *file = fopen(path0pipeline.c_str(),"rb");

  char buf[2];
  int good=1;
#define READ(x) (good&=!!(fread(&(x), sizeof(x), 1, file)));
  int nframes;
  READ(nframes);
  printf("nframes = %d\n", nframes);
  std::vector<ScaleBlob*> frameroots;
  std::unordered_map<ScaleBlob*, ScaleBlob*> allblobs;
  std::vector<std::vector<ScaleBlob*>> frameblobs;
  for(int i=0;i<nframes;++i){
    ScaleBlob *rootaddr;
    int   nblobs;
    READ(rootaddr);
    READ(nblobs);
    frameroots.push_back(rootaddr);
    // printf("root %p has %d blobs.\n", rootaddr, nblobs);
    std::vector<ScaleBlob*> blobs;
    for(int i=0;i<nblobs;++i){
      ScaleBlob* p0;
      ScaleBlob* label;
      int nchildren, npred, nsucc;
      READ(label);
      if(!label)continue;
      ScaleBlob *blob = new ScaleBlob();
      READ(nchildren);
      for(int i=0;i<nchildren;++i){
        READ(p0);
        blob->children.push_back((ScaleBlob*)p0);
      }
      READ(npred);
      for(int i=0;i<npred;++i){
        READ(p0);
        blob->pred.push_back((ScaleBlob*)p0);
      }
      READ(nsucc);
      for(int i=0;i<nsucc;++i){
        READ(p0);
        blob->succ.push_back((ScaleBlob*)p0);
      }
      READ(blob->parent);
      READ(blob->mode);
      READ(blob->position);
      READ(blob->shape);
      READ(blob->fshape);
      READ(blob->timestep);
      READ(blob->covariance);
      READ(blob->invCov);
      READ(blob->detCov);
      READ(blob->pdfCoef);
      READ(blob->min);
      READ(blob->max);
      READ(blob->scale);
      READ(blob->n);
      READ(blob->npass);
      READ(blob->peakvalue);

      allblobs[label] = blob;
      blobs.push_back(blob);
    }
    frameblobs.push_back(blobs);
  }
  if(!good){
    fprintf(stderr, "pipeline.load: read error.\n");
    return;
    // exit(0);
  }
  // printf("fixing...\n");
  for(std::pair<ScaleBlob*, ScaleBlob*> elt : allblobs){
    if(!elt.second)continue;
    // printf("parents...");
    // printf("%p; ", elt.second->parent);
    elt.second->parent = allblobs[elt.second->parent];
    // printf("children...");
    for(int i=0;i<elt.second->children.size(); ++i){
      // printf("%d/%d; ", i, elt.second->children.size());
      elt.second->children[i] = allblobs[elt.second->children[i]];
    }
    // printf("preds...");
    for(int i=0;i<elt.second->pred.size(); ++i){
      // printf("%d/%d; ", i, elt.second->pred.size());
      elt.second->pred[i] = allblobs[elt.second->pred[i]];
    }
    // printf("succs...");
    for(int i=0;i<elt.second->succ.size(); ++i){
      // printf("%d/%d; ", i, elt.second->succ.size());
      elt.second->succ[i] = allblobs[elt.second->succ[i]];
    }
  }
  // printf("\nread......\n");
  // printf("all blobs:");
  int i=0;
  for(std::pair<ScaleBlob*, ScaleBlob*> elt : allblobs){
    // printf("%d: %p\n", i, elt.second);
    ++i;
  }
  // printf("pushing\n");
  // printf("frames %d %d\n", nframes, frames.size());
  for(int i=0;i<nframes;i++){
    ArFrameData fd;
    fd.blob      = allblobs[frameroots[i]];
    // printf("%p\n", fd.blob);
    fd.scales    = collect_scales(fd.blob);
    fd.scale_eps = compute_epsilon(fd.scales);
    fd.complete  = true;
    fd.bspblobs = filter.get_bsp(10);
    
    BSPTree<ScaleBlob> *bsptree = &fd.bspblobs;
    for(ScaleBlob *sb : frameblobs[i]){
      bsptree->insert(sb, sb->position);
    }

    // DEBUG

    // for(float f : fd.scales){
    //   printf("scale %.2f;\n", f);
    // }
    // /////
    frames[i] = fd;
  }

  // load paths...

  if(access(path0paths.c_str(), F_OK) == -1)return;
  file = fopen(path0paths.c_str(),"rb");

  int n;
  READ(n);
  printf("read %d paths.\n", n);
  for(int i=0;i<n;++i){
    std::vector<ScaleBlob*> path;
    int pathlen;
    ScaleBlob *blob;
    READ(pathlen);
    for(int i=0;i<pathlen;++i){
      READ(blob);
      path.push_back(allblobs[blob]);
    }
    paths.push_back(path);
  }

  loadTGMM();

  // done.
#undef READ
}


// void ArPipeline::getTGMMOutput(std::string path, int low, int high){
//   printf("load tgmm output\n");
// }
static void print_element_names(xmlNode * a_node)
 {
    xmlNode *cur_node = NULL;

    for (cur_node = a_node; cur_node; cur_node =
         cur_node->next) {
       if (cur_node->type == XML_ELEMENT_NODE) {
          printf("node type: Element, name: %s\n",
               cur_node->name);
       }
       print_element_names(cur_node->children);
    }
 }

static vec3 from_string(char *in){
  char *token;
  vec3 ret(0,0,0);
 
  token = strtok(in, " ");
  ret.x = std::stof(token);
  token = strtok(NULL, " ");
  ret.y = std::stof(token);
  token = strtok(NULL, " ");
  ret.z = std::stof(token);
  // printf("token = %s", token);
  return ret;
}
static dmat3x3 dmat3_from_string(char *W){
  return mat3x3(5,0,0, 0,5,0, 0,0,5);
}
void ArPipeline::loadTGMM(){
  xmlDoc *doc = NULL;
  xmlNode *root = NULL;


  // map of blobs between frames
  std::map<int, ScaleBlob*> maps[] = {
    std::map<int, ScaleBlob*>(),
    std::map<int, ScaleBlob*>()
  };int mapi = 0;

  for(int i=exp->low, ii=0; i<=exp->high;i++, ii++){
    // printf("TGMM load %s.\n", exp->gettgmmpath(i).c_str());
    doc = xmlReadFile(exp->gettgmmpath(i).c_str(), NULL, 0);
    if(!doc){
      printf("could not read file.\n");
      continue;
    }

    root = xmlDocGetRootElement(doc);


    xmlNode *cur = 0;
    
    frames[ii].tgmm_blob = new ScaleBlob();
    maps[mapi].clear();

    for(cur = root->children; cur; cur = cur->next){
      if (cur->type == XML_ELEMENT_NODE){
        // GaussianMixtureModel
        ScaleBlob *blob = new ScaleBlob();
        xmlChar *m = xmlGetProp(cur, (xmlChar*)"m");
        xmlChar *W = xmlGetProp(cur, (xmlChar*)"W");
        xmlChar *s_id = xmlGetProp(cur, (xmlChar*)"id");
        xmlChar *s_parent = xmlGetProp(cur, (xmlChar*)"parent");

        int id = std::stof((char*) s_id);
        int pred = std::stof((char*) s_parent);

        maps[mapi][id] = blob;
        if(pred != -1){
          maps[mapi][id]->pred.push_back(maps[!mapi][pred]);
          maps[!mapi][pred]->succ.push_back(maps[mapi][id]);
        }
        // printf("m=%s\n", m);
        // printf("W=%s\n", W);
        blob->n = 100;
        blob->scale = 0;
        blob->position = from_string((char*)m);
        blob->min = blob->position - dvec3(10,10,10);
        blob->max = blob->position + dvec3(10,10,10);
        blob->shape = dmat3_from_string((char*)W);
        // printf("m = %.2f %.2f %.2f\n", blob->position.x, blob->position.y, blob->position.z);
        frames[ii].tgmm_blob->children.push_back(blob);
      }
    }
    mapi = !mapi;
    // printf("size[%d] = %p = %d\n", ii, frames[ii].tgmm_blob, frames[ii].tgmm_blob->children.size());
    // print_element_names(root);
    xmlFreeDoc(doc);       // free document
    xmlCleanupParser();    // Free globals
  }

}