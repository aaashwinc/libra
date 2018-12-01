#include <SFML/Graphics.hpp>
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>


#include "view.h"

static inline int ftoi(float i){
  return int(i*255.999999);
}
static inline float itof(int i){
  return float(i)/255.999999;
}
static inline float sq(float in){
  return in*in;
}
vec4 Colormap::colorof(double x){
  // return computecolor(float(x));
  int n = int(x/step);
  if(n<0)n=0;
  if(n>=nsamples)n=nsamples-1;
  return range[n];
}
static inline void put_color(float x, Colormap &colormap, sf::Uint8 *color){
  if(x>=1)x=1;
  if(x<0)x=0;
  vec4 col = colormap.colorof(x);
  color[0] = ftoi(x);
  color[1] = ftoi(x*x);
  color[2] = ftoi(x*x*x);
  color[3] = 255;

}

View::View(int w, int h) : w(w), h(h){
  texture.create(w,h);
  texdata  = new sf::Uint8[w*h*4];
  sprite.setTexture(texture);
  camera   = Camera();
  colormap = Colormap();
  unstable = 1;
  beat     = 0;
}

void View::setvolume(Nrrd *n){
  if(vcache.n == n){
    return;
  }
  NrrdAxisInfo *a = n->axis;

  vcache.n = n;
  vcache.a = n->axis;
  vcache.data = (float*)n->data;
  vcache.w0 = 1;
  vcache.w1 = a[0].size * vcache.w0;
  vcache.w2 = a[1].size * vcache.w1;
  vcache.w3 = a[2].size * vcache.w2;

  vcache.a1 = a[1].size;
  vcache.a2 = a[2].size;
  vcache.a3 = a[3].size;
  touch();
}

float View::qsample(int c, float x, float y, float z){
  int i0 = c;
  int i1 = x;
  int i2 = y;
  int i3 = z;
  if(i1<0 || i2<0 || i3<0 || i1 >= vcache.a1 || i2 >= vcache.a2 || i3 >= vcache.a3){
    return 0.f;
  }
  return vcache.data[
    i3*vcache.w3 + i2*vcache.w2 + i1*vcache.w1 + i0
  ];
}

void View::drawflat(){
  if(camera.flat.slice>=1)camera.flat.slice=1;
  if(camera.flat.slice<0)camera.flat.slice=0;
  float zz=camera.flat.slice;
  int i=0;
  float xx=0,yy=0, vv=0;
  for(int x=0;x<w;++x){
    for(int y=0;y<h;++y){
      xx = float(x)/w * (float(vcache.a1)-0.00005f);
      yy = float(y)/h * (float(vcache.a2)-0.00005f);
      zz = camera.flat.slice * (float(vcache.a3)-0.00005f);
      vv = qsample(0,zz,xx,yy);
      put_color(vv, colormap, texdata+i);
      i+=4;
    }
  }
  texture.update(texdata);
}
void View::raytrace(){
  vec3 forward = camera.look;
  vec3 right   = camera.right;
  vec3 up      = camera.up;

  vec3 ray = forward;
  vec3 dx  = (right * camera.lhorz)/(float(w)/2.f);
  vec3 dy  = -(up   * camera.lvert)/(float(h)/2.f);
  vec3 topleft = forward - (right*camera.lhorz) + (up*camera.lvert);

  // printf("raytrace, %.2f %.2f %.2f + %.2f %.2f %.2f\n",camera.pos.x, camera.pos.y, camera.pos.z, forward.x,forward.y,forward.z);

  int i=0;

  vec3 p;
  float color_x,color_y,color_z,color_w;
  vec4 probe;
  vec3 startp = camera.pos * 33.f;
  for(int py=0;py<h;++py){
    for(int px=0;px<w;++px){
      if(beat < 1){
        ray = topleft + float(px)*dx + float(py)*dy;

        ray = ray * 0.033f * 33.f;
        p   = startp;

        color_x = 0;
        color_y = 0;
        color_z = 0;
        color_w = 1;

        while(color_w > 0.01f){
          p += ray;
    
          float v = qsample(0, p.x, p.y, p.z);

          probe = colormap.colorof(v);
          color_x += probe.x*probe.w*color_w;
          color_y += probe.y*probe.w*color_w;
          color_z += probe.z*probe.w*color_w;
          color_w += -color_w*probe.w;

          color_w *= 0.995f;           // some light is absorbed or refracted away.
        }
        texdata[i+0] = int(color_x*255.999999);
        texdata[i+1] = int(color_y*255.999999);
        texdata[i+2] = int(color_z*255.999999);
        texdata[i+3] = 255;
      }else{
        // texdata[i+0] /= 2;
        // texdata[i+1] /= 2;
        // texdata[i+2] /= 2;
      }
      ++beat;
      if(beat==3)beat = 0;
      i+=4;
    }
  }
  beat += 1;
  beat %= 3;
  texture.update(texdata);
}
void View::touch(){
  unstable = 5;
}
int View::render(){
  if(unstable<=0)return (unstable = 0);
  if(camera.drawflat){
    drawflat();
  }else{
    raytrace();
  }
  --unstable;
  return 1;
}

void View::move3D(vec3 v){
  touch();
  // printf("move: %.2f %.2f %.2f\n",v.x,v.y,v.z);
  // printf("camera: %.2f %.2f %.2f\n",camera.right.x,camera.right.y,camera.right.z);
  camera.pos += v.x*camera.right + v.y*camera.up + v.z*camera.look;
}
void View::rotateH(float r){
  touch();
  camera.set(camera.pos, camera.look - camera.right*r, camera.sky);
}
void View::rotateV(float r){
  touch();
  float dot = glm::dot(camera.look,camera.sky);
  if((r < 0 && dot>-0.9) || (r > 0 && dot < 0.9)){
    camera.set(camera.pos, camera.look + camera.up*r, camera.sky);
  }
}
sf::Sprite &View::getSprite(){
  return sprite;
}
void render_to(sf::Window *window){

}