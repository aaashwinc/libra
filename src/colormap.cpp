#include "view.h"
#include "colormap.h"

Colormap::Colormap(){
  nsamples = 512;
  domain = new float[nsamples];
  range  = new vec4[nsamples];
  step = 2.0/double(nsamples);
  for(int i=0;i<nsamples;++i){
    float x = (float(i)/float(nsamples));
    range[i] = computecolor(x*2.0);
  }
}
vec4 Colormap::colorof(double x){
  int n = int(x/step);
  if(n<0)n=0;
  if(n>=nsamples)n=nsamples-1;
  return range[n];
}
static float sq(float x){
  return x*x;
}
vec4 Colormap::computecolor(float x){
  vec4 color(0,0,0,0);
  if(x<=1.f){
    color.x = sqrt(x);
    color.y = x;
    color.z = 1.f-x;
    float w = x;
    color.w = pow(w,2.5);
  }
  else{
    x -= 1.f;
    color.x = sqrt(fmax(1-fabs(2*x),0));
    color.y = sqrt(fmax(1-fabs(2*(x-0.5f)),0));
    color.z = sqrt(fmax(1-fabs(2*(x-1.f)),0));
    color.w = 0.7f;
    // color=vec4(1,1,1,1);
  }
  return color;
}
