#include <SFML/Graphics.hpp>
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>


#include "view.h"
#include "colormap.h"

static inline int ftoi(float i){
  return int(i*255.999999);
}
static inline float itof(int i){
  return float(i)/255.999999;
}
static inline float sq(float in){
  return in*in;
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

View::View(int w, int h) : w(w), h(h), colormap(4.5f), gamma(4.5f){
  texture.create(w,h);
  texdata  = new sf::Uint8[w*h*4];
  camera   = Camera();
  unstable = 1;
  beat     = 0;
  
}

void View::step_gamma(float factor){
  if(factor<=0)return;
  colormap.destroy();
  gamma *= factor;
  colormap = Colormap(gamma);
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
void View::setgeometry(ArGeometry3D* geometry){
  geom = *geometry;
  lines = sf::VertexArray(sf::Lines, geom.lines.size());
}
void View::draw_geometry(){
  float square = min(win.width, win.height);
  float px = (win.width-square)/2.f;
  float py = (win.height-square)/2.f;

  for(int i=0;i<geom.lines.size();i+=2){
    lines[i+0].color = geom.lines_c[i+0];
    lines[i+1].color = geom.lines_c[i+1];

    line3 clipped = camera.to_screen(line3(geom.lines[i]/33.f, geom.lines[i+1]/33.f), ivec2(square, square));

    // vec3 ps0 = camera.to_screen(geom.lines[i+0], ivec2(square, square));
    // vec3 ps1 = camera.to_screen(geom.lines[i+1], ivec2(square, square));

    if(clipped[0].z <= 0 && clipped[1].z <= 0){
      // if both are behind the camera, then discard both.
      lines[i  ].color = sf::Color::Transparent;
      lines[i+1].color = sf::Color::Transparent;
    }
    // else{
    //   if(ps0.z < 0){
    //     vec3 v = ps1 - ps0; // v.z > 0
    //     ps0 = ps0 + v*(-ps0.z/v.z);
    //   }
    //   if(ps1.z < 0){
    //     vec3 v = ps0 - ps1; // v.z > 0
    //     ps1 = ps1 + v*(-ps1.z/v.z);
    //   }
    // }

    // transform to screen space:
    lines[i+0].position.x = clipped[0].x + px;
    lines[i+0].position.y = clipped[0].y + py;
    lines[i+1].position.x = clipped[1].x + px;
    lines[i+1].position.y = clipped[1].y + py;

    // printf("line %.3f %.3f -> %.3f %.3f\n", lines[i].position.x, lines[i].position.y, lines[i+1].position.x, lines[i+1].position.y);
  }
  
  // lines[0].position = camera.to_screen(vec3());sf::Vector2f(0,0);
  // lines[1].position = sf::Vector2f(0.5f,0.5f);

}
float View::qsample(int c, float x, float y, float z){
  int i0 = c;
  int i1 = x;
  int i2 = y;
  int i3 = z;
  if(i1<0 || i2<0 || i3<0 || i1 >= vcache.a1 || i2 >= vcache.a2 || i3 >= vcache.a3){
    return -1.f;
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
vec3 View::pixel_to_ray(vec2 v){
  
  float square = min(win.width, win.height);
  float px = (win.width-square)/2.f;
  float py = (win.height-square)/2.f;

  vec2 screen((v.x-px)/square, (v.y-py)/square);

  screen.x = (screen.x-0.5f)*2.f;
  screen.y = (screen.y-0.5f)*2.f;

  // vec3 left = -camera.right*camera.lhorz;
  // vec3 up   = camera.up*camera.lvert;

  // vec3 topleft = forward - (right*camera.lhorz) + (up*camera.lvert);

  printf("clicked %.2f %.2f\n",screen.x, screen.y);
  return vec3(camera.look + camera.right*screen.x*camera.lhorz - camera.up*screen.y*camera.lvert);
  // return camera.look;
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

        ray = ray * 0.033f * 33.f * 0.5f;
        p   = startp;

        color_x = 0;
        color_y = 0;
        color_z = 0;
        color_w = 1;

        if(p.x < 0)p = p + ray*(-p.x/ray.x);
        if(p.y < 0)p = p + ray*(-p.y/ray.y);
        if(p.z < 0)p = p + ray*(-p.z/ray.z);

        if(p.x > vcache.a1)p = p - ray*(p.x - vcache.a1)/ray.x;
        if(p.y > vcache.a2)p = p - ray*(p.y - vcache.a2)/ray.y;
        if(p.z > vcache.a3)p = p - ray*(p.z - vcache.a3)/ray.z;

        while(color_w > 0.01f){
          p += ray;
    
          float v = qsample(0, p.x, p.y, p.z);
          if(v<0)break;

          probe = colormap.colorof(v);
          if(probe.w > 0.01f){
            color_x += probe.x*probe.w*color_w;
            color_y += probe.y*probe.w*color_w;
            color_z += probe.z*probe.w*color_w;
            color_w += -color_w*probe.w;
          }

          color_w -= 0.001; // some light is absorbed or refracted away.
          
          // color_w *= 0.995f;           
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
  draw_geometry();
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
void View::render_to(sf::RenderWindow *window){
  static sf::VertexArray quad(sf::TriangleFan, 4);

  sf::Vector2u win_s = window->getSize();

  if(win_s.x != win.width || win_s.y != win.height){
    win.width  = win_s.x;
    win.height = win_s.y;
    draw_geometry();
  }

  float square = min(win_s.x, win_s.y);
  float px = (win_s.x-square)/2.f;
  float py = (win_s.y-square)/2.f;

  quad[0].position = sf::Vector2f(px,py);
  quad[1].position = sf::Vector2f(px+square,py);
  quad[2].position = sf::Vector2f(px+square, py+square);
  quad[3].position = sf::Vector2f(px,py+square);

  quad[0].texCoords = sf::Vector2f(0,0);
  quad[1].texCoords = sf::Vector2f(w,0);
  quad[2].texCoords = sf::Vector2f(w,h);
  quad[3].texCoords = sf::Vector2f(0,h);

  window->draw(quad, &texture);
  window->draw(lines);
}