#include "filter.h"
// #include "medianfilter.h"
#include "HuangQS.h"
#include <vector>
#include <unistd.h>

#define pi (3.14159265358979323846264)
#define gaus(x,stdv) (exp(-((x)*(x))/(2*(stdv)*(stdv)))/((stdv)*sqrt(2*pi)))
#define gausd2(x,sig)(exp(-x*x/(2.0*sig*sig))*(x*x-sig*sig) /       \
     (sig*sig*sig*sig*sig*2.50662827463100050241))

static inline int circle_next(int i, int max){
  i+=1;
  if(i>=max)i=0;
  return i;
}
static inline int circle_prev(int i, int max){
  i-=1;
  if(i<0)i=max-1;
}

//todo: check
inline void Filter::conv1d(Nrrd *nin, Nrrd *nout, int start, int skip, int n, DiscreteKernel kernel){


}
void Filter::conv2d(short *in, short *out, int xlen, int ylen, int zlen, int xstep, int ystep, int zstep, DiscreteKernel kernel){
  // printf("conv2d %d %d\n",xlen, ylen);
  
  int skip  = zstep;
  int n = zlen;

  for(int x=0; x<xlen; ++x){
      // printf("%d\n",x);
    for (int y=0; y<ylen; ++y){
      int start = x*xstep + y*ystep;

      // printf("conv1d %d %d %d\n", start, skip, n);
      // Filter::print_kernel(kernel);

      // printf("input:\n ");
      // for(int i=0;i<n;i++)printf(" %d ",in[start+skip*i]);
      // printf("\n");

      double v = 0;
      for(int i=0;i<n-kernel.radius*2;++i){
        v=0;
        for(int j=0,ji=start+skip*i;j<kernel.support;++j,ji+=skip){
          v += kernel.data[j] * in[ji];
          // printf(" + %.1f*%d", kernel.data[j], in[ji]);
        }
        out[start+skip*i] = short(v);
        // printf(" = %.1f\n",v);
      }

      // printf("output:\n ");
      // for(int i=0;i<n;i++)printf(" %d ",out[start+skip*i]);
      // printf("\n");

      // exit(0);
    }
  }
}
void Filter::filter(){
  short *in  = self.buff[self.curr];
  short *out = self.buff[itempbuf()];

  conv2d(in,  out, self.a1, self.a2, self.a3, self.w1, self.w2, self.w3, self.kernel);  // xy(z) axis.
  conv2d(out, out, self.a1, self.a3, self.a2, self.w1, self.w3, self.w2, self.kernel);  // x(y)z axis.
  conv2d(out, out, self.a2, self.a3, self.a1, self.w2, self.w3, self.w1, self.kernel);  // (x)yz axis.

  int from,to;

  for(int z=self.a3-1;z>=0;z--){
    for(int y=self.a2-1;y>=0;y--){
      for(int x=self.a1-1;x>=0;x--){
        to = x*self.w1 + y*self.w2 + z*self.w3;
        
        if(  x<self.kernel.radius || y<self.kernel.radius || z<self.kernel.radius
          || x>=self.a1-self.kernel.radius || y>=self.a2-self.kernel.radius || z>=self.a3-self.kernel.radius){
            out[to] = 0;
          continue;
        }
        
        from = (x-self.kernel.radius)*self.w1 + (y-self.kernel.radius)*self.w2 + (z-self.kernel.radius)*self.w3;
        out[to] = out[from];
      }
    }
  }
  self.curr = itempbuf();
}
DiscreteKernel Filter::gaussian(double sigma, int radius, int d){
  printf("create gaussian kernel: %.2f, %d\n",sigma, radius);
  DiscreteKernel k;
  k.radius  = radius;
  k.support = radius*2 + 1;
  k.data = new double[k.support];
  k.temp = new double[k.support];
  int h = radius;
  if(d==2)  k.data[h] = gausd2(0,sigma);
  else      k.data[h] = gaus(0,sigma);
  for(int i=1;i<radius+1;++i){
    if(d==2)  k.data[h+i] = gausd2(i,sigma);
    else      k.data[h+i]   = gaus(i,sigma);
    k.data[h-i] = k.data[h+i];
  }
  print_kernel(k);
  return k;
}
DiscreteKernel Filter::laplacian(){
  DiscreteKernel k;
  k.radius = 1;
  k.support = 3;
  k.data = new double[3];
  k.temp = new double[3];
  k.data[0] = 1.0;
  k.data[1] = -2.0;
  k.data[2] = 1.0;
  return k;
}
void Filter::set_kernel(DiscreteKernel k){
  this->self.kernel = k;
}
void Filter::print_kernel(DiscreteKernel k){
  printf("kernel...\n  ");
  for(int i=0;i<k.support;++i){
    printf("%2.3f ",k.data[i]);
  }
  printf("\n/kernel...\n");
}
// struct filter_info{
//   int len;
//   short *in;
//   short *out;
// };
// static filter_info get_info(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info fi;
//   NrrdAxisInfo *a = nin->axis;
//   fi.len = a[0].size * a[1].size * a[2].size *a[3].size;
//   fi.in = (short*)nin->data;
//   fi.out  = (short*)nout->data;
//   return fi;
// }
// void Filter::positive(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info f = get_info(nin, nout, channel);
//   for(int i=channel;i<f.len;i+=2){
//     if(f.in[i]>0)f.out[i]=f.in[i];
//     else f.out[i] = 0;
//   }
// }
// void Filter::negative(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info f = get_info(nin, nout, channel);
//   for(int i=channel;i<f.len;i+=2){
//     if(f.in[i]<0)f.out[i]= f.in[i];
//     else f.out[i] = 0;
//   }
// }
// void Filter::binary(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info f = get_info(nin, nout, channel);
//   for(int i=channel;i<f.len;i+=2){
//     if(f.in[i]>0)f.out[i]=100;
//     else f.out[i] = 0;
//   }
// }
void Filter::laplacian3d(){
  short *in  = self.buff[self.curr];
  short *out = self.buff[itempbuf()];

  printf("dims %d %d %d %d\n",self.a0,self.a1,self.a2,self.a3);
  printf("max %d\n",self.w4);
  int xo;
  for(int z=0;z<self.a3; z++){
    for(int y=0; y<self.a2; y++){
      for(int x=0; x<self.a1; x++){
        xo = x*self.w1 + y*self.w2 + z*self.w3;

        if(x<3 || y<3 || z<3 || x>=self.a1-3 || y>=self.a2-3 || z>=self.a3-3){
          out[xo]=0;
          continue;
        }
        
        int p1=0,p2=0,p3=0,
            p4=0,p5=0,p6=0;
        double v;

        int xi = xo;
        p1 = xi-self.w1;
        p2 = xi+self.w1;
        p3 = xi-self.w2;
        p4 = xi+self.w2;
        p5 = xi-self.w3;
        p6 = xi+self.w3;

        double l1=0,l2=0,l3=0;

        l1 = in[xi] - 0.5*in[p1] - 0.5*in[p2];
        l2 = in[xi] - 0.5*in[p3] - 0.5*in[p4];
        l3 = in[xi] - 0.5*in[p5] - 0.5*in[p6];
        // double mx = fmax(l1,fmax(l2,l3));
        // double mn = fmin(l1,fmin(l2,l3));
        // printf("%f %f %f -> %f %f\n",l1,l2,l3,mx,mn);
        v = (l1+l2+l3)/3.0;
        if(v<0)v=0;
        out[xo] = short(v);
      }
    }
  }
  self.curr = itempbuf();
}

void Filter::threshold(int min, int max){
  short *data = self.buff[self.curr];
  for(int i=0;i<self.w4;i++){
    if(data[i]<min)data[i]=0;
    if(data[i]>max)data[i]=0;
  }
}
void Filter::normalize(double power){
  int channel =0 ;
  
  int buckets[10];
  for(int i=0;i<10;++i)buckets[i] = 0;

  // printf("axes %d %d %d %d\n",self.a0,self.a1,self.a2,self.a3);

  NrrdAxisInfo *a = self.a;
  short *data = self.buff[self.curr];
  short *out  = self.buff[itempbuf()];

  double max = 0;
  double min = 30000;
  for(int i=channel;i<self.w4;i+=2){
    if(data[i] < min)min=data[i];
    if(data[i] > max)max=data[i];
  }
  max = max-min;
  for(int i=channel;i<self.w4;i+=2){
    double r = double(data[i]-min)/double(max);
    if(power == 1)out[i] = r * 30000.0;
    else out[i] = pow(r,power) * 30000.0;
    int bucket = (int)(r*10.0);
    if(bucket<0)bucket=0;
    if(bucket>9)bucket=9;
    buckets[bucket]++;
  }
  printf("histogram:\n ");
  for(int i=0;i<10;i++){
    printf("%4d, ",buckets[i]);
  }
  printf("\n");
  printf("minmax: %.1f %.1f\n",min,min+max);
  self.curr = itempbuf();
}
void Filter::median1(){
  short *in = self.buff[self.curr];
  short *out = self.buff[itempbuf()];
  NrrdAxisInfo *a = self.a;

  int a1 = a[1].size;
  int a2 = a[2].size;
  int a3 = a[3].size;

  short *bufin  = new short[a1*a2*a3];
  short *bufout = new short[a1*a2*a3];

  // short *bufin  = (short*) malloc(sizeof(short)*((a->size[1])*(a->size[2])*(a->size[3])));
  // short *bufout = (short*) malloc(sizeof(short)*((a->size[1])*(a->size[2])*(a->size[3])));
  int dims[3]  = {a1,a2,a3};
  int fmin[3]  = {0,0,0};
  int fsiz[3] = {a1,a2,a3};
  int fmax[3] = {a1,a2,a3};
  int l = a1*a2*a3;
  for(int i=0;i<l;i++){
    bufin[i] = in[i*2];
  }
  median_filter_3D<1>(bufin, dims, bufout, fmin, fsiz, fmax);
  for(int i=0;i<l;i++){
    out[i*2] = bufout[i];
  }
  free(bufin);
  free(bufout);
  self.curr = itempbuf();
}

void Filter::init(Nrrd *nin){
  if(nin == 0){
    self.nrrd = 0;
    self.buff = 0;
    self.nbuf = 0;
    self.curr = 0;
    self.a = 0;

    self.kernel.data = 0;
    self.kernel.temp = 0;
    self.alive = false;
    return;
  }
  if(self.alive){
    destroy();
  }
  self.a = nin->axis;
  
  self.a0=self.a[0].size;
  self.a1=self.a[1].size;
  self.a2=self.a[2].size;
  self.a3=self.a[3].size;

  self.w0=1;     // offset to crawl channel
  self.w1=self.a0*self.w0; // offset to crawl x
  self.w2=self.a1*self.w1; // offset to crawl y
  self.w3=self.a2*self.w2; // offset to crawl z
  self.w4=self.a3*self.w3; // length of entire dataset.

  self.nbuf = 2;
  self.nrrd = new Nrrd* [self.nbuf];
  self.buff = new short*[self.nbuf];
  self.nrrd[0] = nin;
  self.buff[0] = (short*)nin->data;

  for(int i=1;i<self.nbuf;++i){
    self.nrrd[i] = nrrdNew();
    nrrdCopy(self.nrrd[i],nin);
    self.buff[i] = (short*)self.nrrd[i]->data;
    self.curr = 0;
  }
  self.alive = true;
}
Nrrd* Filter::commit(){
  printf("Commit.\n");
  if(0 != self.curr){
    nrrdCopy(self.nrrd[0],self.nrrd[self.curr]);
  }
  return self.nrrd[0];
}
void Filter::destroy(){
  printf("Destroy.\n");
  for(int i=1;i<self.nbuf;++i){
    nrrdNuke(self.nrrd[i]);
  }
  if(self.kernel.data)delete[] self.kernel.data;
  if(self.kernel.temp)delete[] self.kernel.temp;
  init(0);
}
int Filter::itempbuf(int c){
  c++;
  if(c>=self.nbuf){
    return 0;
  }
}
int Filter::itempbuf(){
  return itempbuf(self.curr);
}
Filter::Filter(){
  init(0);
}


std::vector<glm::ivec3> Filter::find_maxima(){
  std::vector<glm::ivec3> maxima;

  short *data = self.buff[self.curr];

  for(int z=0;z<self.a3; z++){
    for(int y=0; y<self.a2; y++){
      for(int x=0; x<self.a1; x++){

        if(x<1 || y<1 || z<1 || x>=self.a1-1 || y>=self.a2-1 || z>=self.a3-1){
          continue;
        }

        int xi = x*self.w1 + y*self.w2 + z*self.w3;
        int p1=0,p2=0,p3=0,
            p4=0,p5=0,p6=0, v;

        v  = data[xi];
        p1 = data[xi-self.w1];
        p2 = data[xi+self.w1];
        p3 = data[xi-self.w2];
        p4 = data[xi+self.w2];
        p5 = data[xi-self.w3];
        p6 = data[xi+self.w3];

        if(v>=p1 && v>=p2 && v>=p3 && v>=p4 && v>=p5 && v>=p6){
          maxima.push_back(glm::ivec3(x,y,z));
        }
      }
    }
  }
  return maxima;
}
void Filter::highlight(std::vector<glm::ivec3> points){
  printf("maxima: %d\n",points.size());
  int max = 0;
  
  short *buff = self.buff[self.curr];

  for(int i=0;i<self.w4;i+=self.w1){
    if(buff[i]>max)max=buff[i];
    buff[i] = (buff[i]*3)/4;
  }
  for(int i=0;i<points.size();++i){
    glm::ivec3 v = points[i];
    int xi = v.x*self.w1 + v.y*self.w2 + v.z*self.w3;
    buff[xi] = max;
  }
}

// given an input image, find all blobs.
// a blob is a local maximum surrounded by
// its hinderland. ie.:
// a pixel p is in a blob B (centered at pixel b) if 
// p.climb.climb. .... .climb = b.
//
// where p.climb is the adjacent pixel to p with greatest value.
// and b.climb = b.
// we describe this climbing action like a "balloon" which reaches
// its "apex".

typedef unsigned int uint;

struct Balloon{
  uint gradient; // offset to higher balloon (or 0)
  uint volume;   // number of voxels which lead here.
  glm::ivec3 position;
};

std::vector<ScaleBlob*> Filter::find_blobs(){
  using namespace glm;

  // maintain a list of all blobs. return this list.
  std::vector<ScaleBlob*> blobs;
  short *data = self.buff[self.curr];

  // init balloons to 0.
  Balloon *balloons = new Balloon[self.w4/self.w1];
  for(int i=0;i<self.w4/self.w1;i++){
    balloons[i].gradient = 0;
    balloons[i].volume   = 0;
    balloons[i].position = ivec3(0,0,0);
  }

  int x=0,y=0,z=0;
  for(int i=0;i<self.w4/self.w1;i++){

    short v = data[2*i];
    
    // get values of neighbors.
    int offneighs[6]  = {0,0,0,0,0,0};

    // get values of neighbors with out-of-bounds checking.
    if(x>0)         offneighs[0] = -self.w1 / self.w1;  // divide by w1 because
    if(x<self.a1-1) offneighs[1] =  self.w1 / self.w1;  // we're ignoring the
    if(y>0)         offneighs[2] = -self.w2 / self.w1;  // first [channel] axis.
    if(y<self.a2-1) offneighs[3] =  self.w2 / self.w1;
    if(z>0)         offneighs[4] = -self.w3 / self.w1;
    if(z<self.a3-1) offneighs[5] =  self.w3 / self.w1;
    
    // if(i>1020 && i<1030){
    //   printf("xyz = %d %d %d\n",x,y,z);
    //   printf("offneighs = %d %d %d %d %d %d\n",offneighs[0],offneighs[1],offneighs[2],offneighs[3],offneighs[4],offneighs[5]);
    // }
    // get maximum neighbor.
    int  max_neighbor = 0;
    bool peak = true;

    for(int j=0;j<6;++j){
      if(data[2*(i+offneighs[j])] >= data[2*(i+max_neighbor)]){
        max_neighbor = offneighs[j];
        peak = false;
      }
    }
    // if(i>1020 && i<1030)printf("max: data[%d+%d] = %d\n",i, max_neighbor, max_value);
    balloons[i].gradient = (max_neighbor);      // set next higher voxel (or 0).
    balloons[i].position = ivec3(x,y,z);
    if(peak){
      printf("apex at %d %d %d\n",x,y,z);
      printf("%d > %d %d %d %d %d %d\n",data[i],
        data[2*(i+offneighs[0])], data[2*(i+offneighs[1])],
        data[2*(i+offneighs[2])], data[2*(i+offneighs[3])],
        data[2*(i+offneighs[4])], data[2*(i+offneighs[5])]);
    }
    // if(i>1020 && i<1030)printf("balloon[%d].gradient = %d\n",i, balloons[i].gradient);
    // if(max_neighbor == i){
    //   balloons[i].apex = new ScaleBlob;   // this is an apex == Blob.
    //   blobs.push_back(balloons[i].apex);  // keep track of this blob.
    // }
    // next voxel.
    ++x;
    if(x == self.a1){
      x=0;
      ++y;
    }if(y == self.a2){
      y=0;
      ++z;
    }if(z == self.a3){
      z=0;
      break;
    }
  }

  // exit(0);

  // now that we have computed where each balloon goes, we can compute our blobs.
  // printf("debug...\n");
  // printf("balloon[%d].gradient = %d\n",1022, balloons[1022].gradient);

  for(int i=0;i<self.w4/self.w1;i++){
    // printf("i %d\n",i);
    Balloon here = balloons[i];
    Balloon apex = here;
    int j = i;
    while(apex.gradient != 0){
      j = j+apex.gradient;

      if(j<0){
        printf("balloon[%d].gradient = %d\n",j-apex.gradient, apex.gradient);
        printf("less than 0!\n");
      }

      apex = balloons[j];
    }
    balloons[j].volume += 1;
  }
  int ct = 0;
  for(int i=0;i<self.w4/self.w1;i++){
    if(balloons[i].volume > 1){
      ivec3 p = balloons[i].position;
      // printf("apex at %d %d %d\n",p.x,p.y,p.z);
      ++ct;
    }
  }
  printf("total number apexes: %d\n",ct);

  //   a->apex->voxels.push_back(b->position);
  // }
  printf("done.\n");
  // usleep(2000000);
  // exit(0);
  delete[] balloons;
  return blobs;
}