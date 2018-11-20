#include <teem/meet.h>
#include <teem/nrrd.h>
#include <glm/glm.hpp>
#include "synth.h"
#include <vector>

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#define pi (3.14159265358979)
#define gaus(x,stdv) (exp(-((x)*(x))/(2*(stdv)*(stdv)))/((stdv)*sqrt(2*pi)))

#define DIM 100

void synth(){
  return;
  // 10 20 60
  using namespace glm;
  short *data = new short[2*DIM*DIM*DIM];
  Nrrd *nval = nrrdNew();

  srand (time(NULL));

  int a0=DIM,a1=DIM,a2=DIM;

  std::vector<vec3> ps;
  for(int i=0;i<1;i++){
    double x = (rand() % DIM/10)/(DIM/10.0);
    double y = (rand() % DIM/10)/(DIM/10.0);
    double z = (rand() % DIM/10)/(DIM/10.0);
    printf("p %.0f %.0f %.0f\n",x*DIM,y*DIM,z*DIM);
    ps.push_back(vec3(x,y,z));
  }


  double x,y,z,v;
  for(int i=0;i<a0*a1*a2;i++){
    x = (i%(DIM))/double(DIM);
    y = ((i/DIM)%DIM)/double(DIM);
    z = (i/(DIM*DIM))/double(DIM);
    vec3 p(x,y,z);
    v = 0;

    for(int i=0;i<ps.size();i++){
      double d = distance(ps[i],p);
      v += gaus(d,0.07);
      // v += gaus(d,0.8);
      // v = y;
    }

    v /= (ps.size()*gaus(0,0.07));

    if(v<0)v=0;
    if(v>1)v=1;

    data[i*2] = v*30000;
  }
  
  nrrdWrap_va(nval, data, nrrdTypeShort, 4, 2, DIM, DIM, DIM);
  nrrdSave("/home/ashwin/data/synth/000.nrrd", nval, NULL);
  nrrdNix(nval);
  delete[] data;
}