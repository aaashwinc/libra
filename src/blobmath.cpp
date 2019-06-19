#include "blobmath.h"

#include <unordered_map>
#include <deque>
#include <set>

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

/*
 * find the longest paths (in order from longest to shortest) in a given list
 * of ScaleBlobs (and their children). Do this by repeating:
 *   1. using dynamic programming to find the longest path in the graph,
 *      maintaining a memoized longest path for each ScaleBlob/node.
 *   2. removing the longest path, and then updating the memoized values
 *      for each of the children and predecessors in the path.
 *   -- EXIT when there are no more paths, or the path reaches a threshold.
*/
std::vector<std::vector<ScaleBlob*>> longest_paths(std::vector<ScaleBlob*> input){
  printf("compute longest paths, input size %d\n", input.size());
  struct pathinfo{
    pathinfo(){
      next = 0;
      length = 0;
      alive = true;
      mature = false;
    }
    pathinfo(ScaleBlob *next, int length, bool alive, bool mature) : 
      next(next), length(length), alive(alive), mature(mature){}

    ScaleBlob* next;  // longest path successor.
    int length;       // length of longest path.
    bool alive;       // can we still use this blob as a node? (not been removed?)
    bool mature;
  };

  std::unordered_map<ScaleBlob*, pathinfo> paths; // store pathinfo for each blob.
  std::deque<ScaleBlob*> traverse;                // which blobs do we have to traverse?

  std::vector<std::vector<ScaleBlob*>> return_paths;  // list of paths that we find.
  pathinfo global_longest;                            // keep track of the longest path.

  int loopn = 0;
  while(1){
    // printf("loop %d\n", ++loopn);
    float nblobs = 0;
    float nalive = 0;
    for(std::pair<ScaleBlob *, pathinfo> info : paths){
      nblobs += 1;
      if(info.second.alive)nalive+=1;
    }
    // printf("alive: %.1f / %.1f = %.2f\n", nalive, nblobs, nalive/nblobs);
    // initialize with blobs in input.
    for(int i=0;i<input.size();i++){
      traverse.push_front(input[i]);
    }

    ScaleBlob *longest = 0;

    int loop1n = 0;
    while(!traverse.empty()){
      ScaleBlob *curr = traverse.front();
      if(!paths[curr].alive){
        traverse.pop_front();
        continue;
      }
      ++loop1n;
      pathinfo pi = paths[curr];
      bool ready = true;
      // printf("traverse(%d)\n", traverse.size());
      // printf("%p...");
      for(ScaleBlob *child : curr->succ){
        if(paths[child].alive && !paths[child].mature){
          traverse.push_front(child);
          ready = false;
          // printf("add %p. ",child);
        }
        // printf("\n");
      }
      if(ready){
        // printf("ready.\n");
        // we have computed path lengths for all of
        // our children. Now compute ourselves.
        pathinfo longestnext;
        for(ScaleBlob *succ : curr->succ){
          pathinfo child = paths[succ];
          if(child.alive){
            bool child_longer  = (child.length >  longestnext.length);
            bool child_equal   = (child.length == longestnext.length);
            bool child_smaller = !longestnext.next || (succ->scale   < longestnext.next->scale);
            if(child_longer || (child_equal && child_smaller)) {
              longestnext.next   = succ;
              longestnext.length = child.length;
            }
          }
        }

        // this is the longest path (and its length)
        pi.next   = longestnext.next;
        pi.length = longestnext.length + 1; // one more than our child's.
        pi.mature = true;
        paths[curr] = pi;
        traverse.pop_front();

        // keep track of the longest path.
        if(!longest || paths[curr].length > paths[longest].length
           || ((paths[curr].length == paths[longest].length) && (curr->scale < longest->scale))){
          longest = curr;
        }
        // printf("path %p -> %p (%d)\n", curr, pi.next, pi.length);
        // printf("sealed, popped. %p -> %p (%d).\n",curr, pi.next, pi.length);

      }
    }
    printf("inner loop %d iterations.\n", loop1n);

    // all blobs are dead. there is no longest.
    if(!longest)break;

    // printf("compute fullpath.\n");
    // compute the list representing the full path.
    std::vector<ScaleBlob*> fullpath;
    ScaleBlob *t = longest;
    while(t){
      fullpath.push_back(t);
      t = paths[t].next;
    }

    // printf("kill.\n");
    // kill all points in path and their children; and invalidate all of their predecessors.
    std::deque<ScaleBlob*> invalidate;
    std::deque<ScaleBlob*> kill;

    // printf("kill parents.\n");
    // kill grand-parents of all points in path (but not grand-uncles)
    for(ScaleBlob *sb : fullpath){
      while(sb){
        paths[sb].alive = false;
        invalidate.push_back(sb);
        sb = sb->parent;
      }
    }
    for(ScaleBlob *sb : fullpath){
      kill.push_back(sb);
    }
    // printf("kill: \n");
    // for(ScaleBlob *sb : kill){
    //   printf("%p ", sb);
    // }
    // printf("\n");

    // printf("kill children.\n");
    // kill children all points in path.
    int nk = 0;
    while(!kill.empty()){
      // printf("s %d\n", kill.size());
      ScaleBlob *curr = kill.front();
      // printf("c %p\n", curr);
      kill.pop_front();
      // printf("alive = false\n");
      paths[curr].alive = false;
      ++nk;
      // printf("invalidate += curr\n");
      invalidate.push_back(curr);
      // printf("~~~\n");
      // printf("cs: %d\n", curr->children.size());
      for(ScaleBlob *child : curr->children){
        // printf("kill.push_back(%p)\n", child);
        kill.push_back(child);
      }
    }
    printf("path len %d; killed children %d\n", fullpath.size(), nk);

    // printf("invalidate.\n");
    // invalidate all predecessors of all points killed.
    while(!invalidate.empty()){
      ScaleBlob *curr = invalidate.front();
      invalidate.pop_front();
      if(paths[curr].mature){
        paths[curr].mature = false;
        for(ScaleBlob *child : curr->pred){
          if(paths[child].mature){
            invalidate.push_back(child);
          }
        } 
      }
    }
    return_paths.push_back(fullpath);
    // return return_paths;
  }

  return return_paths;
}


void serialize(std::vector<ScaleBlob*> blobs, FILE *file){
  std::deque<ScaleBlob*> traverse;
  std::set<ScaleBlob*> allblobs;

  for(ScaleBlob *sb : blobs){
    traverse.push_back(sb);
  }
  while(!traverse.empty()){
    ScaleBlob *curr = traverse.front();
    if(allblobs.find(curr) != allblobs.end()){
      // not already collected. collect blob and all of its relatives.
      for(ScaleBlob *sb : curr->children){
        traverse.push_back(sb);
      }
      for(ScaleBlob *sb : curr->pred){
        traverse.push_back(sb);
      }
      for(ScaleBlob *sb : curr->succ){
        traverse.push_back(sb);
      }
      traverse.push_back(curr->parent);
    }
    traverse.pop_front();
  }



  // allblobs contains all blobs.
  // the format we use is:
  // [addr1] [data1] [addr2] [data2] ...

  for(ScaleBlob *sb : allblobs){

  }

} 


std::vector<ScaleBlob*> deserialize(FILE *file){
  return std::vector<ScaleBlob*>();
}