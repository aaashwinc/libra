#include "view.h"

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
  color.x = sqrt(x);
  color.y = x;
  color.z = 1.f-x;
  // color.z = ftoi(2/(1+(x-0.5)*(x-0.5)) - 1.6);
  float w = x;
  // if(w<0)w=0;
  // w*=1/0.99f;
  // w = w*;
  color.w = pow(w,2.5);
  // color = vec4(x,x,x,x);
  return color;
}
