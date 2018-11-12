#include <SFML/Graphics.hpp>
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>


#include "experiment.h"
#include "view.h"

static inline int ftoi(float i){
  return int(i*255.999999);
}
static inline float itof(int i){
  return float(i)/255.999999;
}

Camera::Camera(){
  lhorz = 1;
  lvert = 1;
  yaw=0,pitch=0;
  pos=vec3();
  look=vec3(), up=vec3(), right=vec3();
  screenToWorld=mat4();
}
void Camera::set(vec3 pos, vec3 look, vec3 up){
  this->pos = pos;
  this->look = normalize(look);
  this->right = normalize(cross(look, up));
  this->up = cross(right, look);
}

Colormap::Colormap(){
  nsamples = 1000;
  domain = new float[nsamples];
  range  = new vec4[nsamples];
  step = 1.0/double(nsamples);
  for(int i=0;i<nsamples;++i){
    float x = (float(i)/float(nsamples));
    range[i] = computecolor(x);
  }
}
vec4 Colormap::computecolor(float x){
  vec4 color(0,0,0,0);
  // color.x = ftoi(1/(1+(x-0.8)*(x-0.8)) -0.6);
  // color.y = ftoi(1/(1+4*(1-x)*(1-x)) -0.2);
  color.x = x;
  color.y = x*x;
  color.z = 1.f-x;
  // color.z = ftoi(2/(1+(x-0.5)*(x-0.5)) - 1.6);
  float w = x-0.2;
  if(w<0)w=0;
  w*=1/0.8f;
  w = w*w*w*w;
  color.w = w;
  // color = vec4(x,x,x,x);
  return color;
}
inline vec4 Colormap::colorof(double x){
  // return computecolor(float(x));
  int n = int(x/step);
  if(n<0)n=0;
  if(n>=nsamples)n=nsamples-1;
  return range[n];
}

View::View(int w, int h) : w(w), h(h){
  mode = 1;
  texture.create(w,h);
  texdata = new sf::Uint8[w*h*4];
  sprite.setTexture(texture);
  camera = Camera();
  colormap = Colormap();
  position =0;
  timestep =0 ;
}

void View::get_color(float x, sf::Uint8 *color){
  if(x>=1)x=1;
  if(x<0)x=0;
  vec4 col = colormap.colorof(x);
  color[0] = ftoi(col.x);
  color[1] = ftoi(col.y);
  color[2] = ftoi(col.z);
  color[3] = 255;

}
long View::check(){
  long v = 0;
  Nrrd *n = experiment->nrrds[0];
  short *data = (short*)n->data;
  for(int i=0;i<171429210;++i){
    v += data[i];
  }
  return v;
}
inline float View::qsample(int c, float x, float y, float z){

  int i0,i1,i2,i3;

  i0 = c;
  i1 = x;
  i2 = y;
  i3 = z;

  if(i1<0 || i2<0 || i3<0 || i1 >= vcache.a1 || i2 >= vcache.a2 || i3 >= vcache.a3){
    return 0.f;
  }

  int i = i3*vcache.w3 + i2*vcache.w2 + i1*vcache.w1 + i0;
  return vcache.data[i]/3000.f;
}
void View::setvolume(int t){
  Nrrd *n = experiment->nrrds[t];
  NrrdAxisInfo *a = n->axis;

  vcache.n = n;
  vcache.a = n->axis;
  vcache.data = (short*)n->data;
  vcache.w0 = 1;
  vcache.w1 = a[0].size * vcache.w0;
  vcache.w2 = a[1].size * vcache.w1;
  vcache.w3 = a[2].size * vcache.w2;

  vcache.a1 = a[1].size;
  vcache.a2 = a[2].size;
  vcache.a3 = a[3].size;
}
float View::sample(int t, int c, float x, float y, float z, float defaultv, bool normalize){
  // static long big = 0;
  // static long sum = 0;
  Nrrd *n = experiment->nrrds[t];
  NrrdAxisInfo *a = n->axis;
  short *data = (short*)n->data;

  int w0,w1,w2,w3;
  int i0,i1,i2,i3;

  w0 = 1;
  w1 = a[0].size * w0;
  w2 = a[1].size * w1;
  w3 = a[2].size * w2;

  // c = 1;
  i0 = c;
  if(normalize){
    i1 = x * (float(a[1].size)-0.00005f);
    i2 = y * (float(a[2].size)-0.00005f);
    i3 = z * (float(a[3].size)-0.00005f);
  }else{
    i1 = x;
    i2 = y;
    i3 = z;
  }
  if(i1<0 || i2<0 || i3<0 || i1 >= a[1].size || i2 >= a[2].size || i3 >= a[3].size){
    // sum += i1+i2+i3;
    // printf("sum %u\n",sum);
    return defaultv;
  }

  int i = i3*w3 + i2*w2 + i1*w1 + i0*w0;
  // if(i>big){
  //   big=i;
  //   printf("big %u\n",big);
  // }
  float v = data[i]/3000.f;

  return v;
}
void View::render(){
  sf::Clock clock;
  sf::Time elapsed1 = clock.getElapsedTime();
  if(position>=1)position=1;
  if(position<0)position=0;
  if(timestep>2)timestep=2;
  if(timestep<0)timestep=0;
  float zz=position;
  int i=0;
  float xx=0,yy=0, vv=0;
  for(int x=0;x<w;++x){
    for(int y=0;y<h;++y){
      xx = float(x)/float(w);
      yy = float(y)/float(h);
      vv = sample(timestep,0,zz,xx,yy);
      get_color(vv, texdata+i);
      i+=4;
    }
  }
  sf::Time elapsed2 = clock.getElapsedTime();
  std::cout <<"time: "<< (elapsed2.asSeconds() - elapsed1.asSeconds()) << std::endl;

  texture.update(texdata);
}

static inline float sq(float in){
  return in*in;
}

void View::move(vec3 v){
  camera.pos += v;
}
void View::raytrace(){
  

  if(timestep>2)timestep=2;
  if(timestep<0)timestep=0;
  
  vec3 forward = camera.look;
  vec3 right   = camera.right;
  vec3 up      = camera.up;

  vec3 ray = forward;
  vec3 dx  = (right * camera.lhorz)/(float(w)/2.f);
  vec3 dy  = -(up    * camera.lvert)/(float(h)/2.f);
  vec3 topleft = forward - (right*camera.lhorz) + (up*camera.lvert);

  int i=0;

  for(int py=0;py<h;++py){
    for(int px=0;px<w;++px){
      ray = topleft + float(px)*dx + float(py)*dy;
      // ray = normalize(ray) * 0.033f;
      ray = ray * 0.033f;
      vec3 p = camera.pos;
      vec4 color(0,0,0,1), probe(0);
      while(color.w > 0.01f){
        p += ray;
  
        float v = qsample(0, p.x*33.f, p.y*33.f, p.z*33.f);

        probe = colormap.colorof(v);
        color += vec4((probe.x*probe.w*color.w), (probe.y*probe.w*color.w), (probe.z*probe.w*color.w), -color.w*probe.w);
        // color.x = 0;

        // color.w *= (1.f - probe.w); // this light is reflected back and is no longer part of the ray.
        color.w *= 0.993f;           // some light is absorbed or refracted away.
      }
      texdata[i+0] = ftoi(color.x);
      texdata[i+1] = ftoi(color.y);
      texdata[i+2] = ftoi(color.z);
      texdata[i+3] = 255;
      i+=4;
    }
  }

  texture.update(texdata);
}
sf::Sprite &View::getSprite(){
  return sprite;
}
void View::setExperiment(Experiment *e){
  this->experiment = e;
}
