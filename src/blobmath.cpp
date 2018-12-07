#include "blobmath.h"

#include <unordered_map>
#include <deque>

std::vector<ScaleBlob*> longest_path(ScaleBlob* sb){
  // printf("longest path %p\n", sb);
  struct pathinfo{
    ScaleBlob* pathnext;  // longest path successor.
    int pathlength;       // length of longest path.
    bool expanded;        // are the children on the queue?
  };
  // maintain a map, ScaleBlob -> longest path.
  // use dynamic programming.

  std::unordered_map<ScaleBlob*, pathinfo> paths;
  std::deque<ScaleBlob*> traverse;
  traverse.push_front(sb);
  paths[sb] = {0, 0, false};
  while(!traverse.empty()){
    ScaleBlob *curr = traverse.front();
    // printf("traverse (%lu): head = %p\n",traverse.size(), curr);
    pathinfo pi = paths[curr];
    if(!pi.expanded){
      // add all of its children to the beginning of the queue
      // so we traverse them before computing this one.
      for(ScaleBlob *child : curr->succ){
        traverse.push_front(child);
        paths[child] = pathinfo{0, 0, false};
      }
      pi.expanded=true;
      // printf("added %p->children. traverse len = %lu\n",curr, traverse.size());
      paths[curr] = pi;
    }
    else{
      // we have computed path lengths for all of
      // our children. Now compute ourselves.
      pathinfo longest = {0, 0, true};
      for(ScaleBlob *succ : curr->succ){
        pathinfo child = paths[succ];
        if(child.pathlength >= longest.pathlength){
          longest.pathnext   = succ;
          longest.pathlength = child.pathlength;
        }
      }

      // this is the longest path (and its length)
      pi.pathnext = longest.pathnext;
      pi.pathlength = longest.pathlength + 1; // one more than our child's.
      paths[curr] = pi;
      traverse.pop_front();
      // printf("path %p -> %p (%d)\n", curr, pi.pathnext, pi.pathlength);
      // printf("sealed, popped. %p -> %p (%d).\n",curr, pi.pathnext, pi.pathlength);
    }
  }
  std::vector<ScaleBlob*> result;
  ScaleBlob *t = sb;
  // printf("compute vector.\n");
  while(t){
    result.push_back(t);
    t = paths[t].pathnext;
  }
  return result;
}

std::vector<std::vector<ScaleBlob*>> longest_paths(std::vector<ScaleBlob*> sb){
  // struct pathinfo{
  //   ScaleBlob* pathnext;  // longest path successor.
  //   int pathlength;       // length of longest path.
  //   bool expanded;        // are the children on the queue?
  // };
  // std::unordered_map<ScaleBlob*, pathinfo> paths;
  // std::deque<ScaleBlob*> traverse;
  // traverse.push_front(sb);
  // paths[sb] = {0, 0, false};
}