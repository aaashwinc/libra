#ifndef BSPTREE_H
#define BSPTREE_H

#include "blob.h"
#include <glm/glm.hpp>
#include <vector>
#include <algorithm>

using namespace glm;

struct plane{
  vec3 point;
  vec3 normal;
};
template<class T>
class BSPTree{
private:
  struct E{
    T *v;
    vec3 p;
  };
public:
  struct D{
    D(T *v, float d) : v(v), d(d){}
    T *v;
    float d;
  };
  BSPTree(BSPTree *parent, int depth, vec3 min, vec3 max) : parent(parent), min(min), max(max), children(0){
    split(depth);
  }
  
  vec3 min;
  vec3 max;
  plane cutting;
  BSPTree *parent;
  BSPTree *children;      // enforce: array of size either 0 or 2.
  std::vector<E> members;
  int n;                  // number of children = 

  std::vector<T*> find_within_distance(vec3, float);

  T *find_closest(vec3 q){
    T* ret = 0;
    float dist = 0;
    find_closest(q, ret, dist);
    return ret;
  }
  void find_closest(vec3 q, T* &v, float &maxdsq){
    if(!v){
      // no candidate point found yet. search for one.
      if(children){
        children[0].find_closest(q, v, maxdsq);
        children[1].find_closest(q, v, maxdsq);
      }else{
        for(E e : members){
          float dsq = distsq(q, e.p);
          if(dsq < maxdsq){
            maxdsq = dsq;
            v = e.v;
          }
        }
      }
    }
    // if q is within the appropriate distance to this cell...
    if(distsq_point_bbox(q) < maxdsq){
      if(children){
        children[0].find_closest(q, v, maxdsq);
        children[1].find_closest(q, v, maxdsq);
      }else{
        for(E e : members){
          float dsq = distsq(q, e.p);
          if(dsq < maxdsq){
            maxdsq = dsq;
            v = e.v;
          }
        }
      }
    }
  }
  // distance between p and this BSPTree's bounding box (given by min/max)
  float distsq_point_bbox(vec3 p){
    vec3 q(p);  // closest point within box to p.
    
    // constrain q to be within the box.
         if(q.x<min.x)q.x = min.x;
    else if(q.x>max.x)q.x = max.x;
    
         if(q.y<min.y)q.y = min.y;
    else if(q.y>max.y)q.y = max.y;
    
         if(q.z<min.z)q.z = min.z;
    else if(q.z>max.z)q.z = max.z;
    
    return distsq(p,q);
  }
  std::vector<T*> as_vector(){
    std::vector<T*> ret;
    ret.insert(ret.end(), members.begin(), members.end());
    if(children){
      std::vector<E> lc = children[0]->as_vector();
      std::vector<E> rc = children[1]->as_vector();
      ret.insert(ret.end(), lc.begin(), lc.end());
      ret.insert(ret.end(), rc.begin(), rc.end());
    }
    return ret;
  }

  void clear(){
    members.clear();
    if(children){
      children[0].clear();
      children[1].clear();
    }
  }
  void split(int depth){
    if(depth == 0)return;

    children = new BSPTree[2];

    float dx = max.x - min.x;
    float dy = max.y - min.y;
    float dz = max.z - min.z;

    vec3 center = 0.5f*(min+max);

    cutting.point = center; // splitting plane intersects center.
    
    if(dx >= dy && dx >= dz)      cutting.normal = vec3(1,0,0);
    else if(dx >= dy && dx >= dz) cutting.normal = vec3(0,1,0);
    else if(dx >= dy && dx >= dz) cutting.normal = vec3(0,0,1);

    children[0] = BSPTree(this, depth-1, min, max - cutting.normal*(dx/2.f));
    children[1] = BSPTree(this, depth-1, min + cutting.normal*(dx/2.f), max);

    for(int i=0;i<members.size();i++){
      insert(members[i].v, members[i].p);
    }
    members.clear();
  }
  static float distsq(vec3 a, vec3 b){
    vec3 v = a-b;
    return v.x*v.x + v.y*v.y + v.z*v.z;
  }

  BSPTree *getsubtree(vec3 p){
    if(dot(p - cutting.point,cutting.normal) < 0){
      return children;
    }else{
      return children+1;
    }
  }
  void insert(T *v, vec3 p){
    if(!children){
      E elt;
      elt.v = v;
      elt.p = p;
      members.push_back(elt);
      return;
    }
    getsubtree(p)->insert(v,p);
  }
  bool remove(T *v){
    for(int i=0;i<members.size();i++){
      if(v[i] == v){
        members.erase(members.begin()+i);
        return true;
      }
    }
    if(children && children[0]->remove(v))return true;
    if(children && children[1]->remove(v))return true;
    return false;
  }
  bool contains(T *v){
    if(std::find(members.begin(), members.end(),v) != members.end())return true;
    if(children && children[0]->contains(v))return true;
    if(children && children[1]->contains(v))return true;
    return false;
  }
};

#endif