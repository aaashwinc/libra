#include <teem/meet.h>
#include <teem/nrrd.h>
#include <glm/glm.hpp>
#include "synth.h"
#include <vector>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "blob.h"

#define pi (3.14159265358979)
#define gaus(x,stdv) (exp(-((x)*(x))/(2*(stdv)*(stdv)))/((stdv)*sqrt(2*pi*pi*pi*2*2)))

#define DIM 100

// problems with >=85 and 10/20

void synth(){
  // return;
  // 10 20 60
  using namespace glm;
  int a0=DIM,a1=DIM,a2=DIM;
  short *data = new short[a0*a1*a2];
  Nrrd *nval = nrrdNew();

  srand ((unsigned long long)nval);


  std::vector<vec3> ps;
  for(int i=0;i<10;i++){
    double x = (rand() % a0/10)/(a0/10.0);
    double y = (rand() % a1/10)/(a1/10.0);
    double z = (rand() % a2/10)/(a2/10.0);
    printf("p %.3f %.3f %.3f\n",x*a0,y*a1,z*a2);
    ps.push_back(vec3(x,y,z));
  }

  ScaleBlob blob;
  blob.position = vec3(50,50,50);
  blob.model.alpha = 1.f;
  blob.model.beta = 0.1;
  blob.model.kappa = 0.8;
  blob.invCov = glm::mat3(0.01f,0,0, 0,0.02f,0, 0,0,0.04f);

  double stdv = 0.125;

  double x,y,z,v;
  for(int i=0;i<a0*a1*a2;i++){
    x = (i%(DIM));
    y = ((i/DIM)%DIM);
    z = (i/(DIM*DIM));
    vec3 p(x,y,z);
    v = 0;

    for(int i=0;i<ps.size();i++){
      vec3 q = p-ps[i];
      // q.z *= 1.f;
      double d = sqrt(q.x*q.x + q.y*q.y + q.z*q.z);
      v += gaus(d,stdv);
      // v += gaus(d,0.8);
      // v = y;
    }

    v /= (ps.size()*gaus(0,stdv));

    // v = 0;

    // printf("%.2f %.2f %.2f\n", p.x, p.y, p.z);

    if(v >= 1){
      printf("v %.3f\n", v);
    }

    if(v<0)v=0;
    if(v>1)v=1;

    v = blob.erf_pdf(p);

    data[i] = v*30000;
  }
  data[0] = 0;
  data[1] = 30000;
  printf("dims %d %d %d\n", a0, a1, a2);
  nrrdWrap_va(nval, data, nrrdTypeShort, 3, a0,a1,a2);
  nrrdSave("/home/ashwin/data/synth/000.nrrd", nval, NULL);
  nrrdSave("/home/ashwin/data/synth/001.nrrd", nval, NULL);
  nrrdSave("/home/ashwin/data/synth/002.nrrd", nval, NULL);
  nrrdSave("/home/ashwin/data/synth/003.nrrd", nval, NULL);
  nrrdNix(nval);
  delete[] data;
}