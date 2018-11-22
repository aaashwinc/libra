#include "filter.h"
// #include "medianfilter.h"
#include "HuangQS.h"
#include <vector>
#include <queue> 
#include <unordered_set>
#include <unordered_map>
#include <unistd.h>
#include <climits>
#define pi (3.14159265358979323846264)
#define gaus(x,stdv) (exp(-((x)*(x))/(2*(stdv)*(stdv)))/((stdv)*sqrt(2*pi)))
#define gausd2(x,sig)(exp(-x*x/(2.0*sig*sig))*(x*x-sig*sig) /       \
     (sig*sig*sig*sig*sig*2.50662827463100050241))


DiscreteKernel::DiscreteKernel(){
  data = 0;
  temp = 0;
}
void Filter::conv2d(short *in, short *out, int xlen, int ylen, int zlen, int xstep, int ystep, int zstep, DiscreteKernel kernel){
  // printf("conv2d %d %d\n",xlen, ylen);
  
  int skip  = zstep;
  int n = zlen;

  // n = 20;

  for(int x=0; x<xlen; ++x){
      // printf("%d\n",x);
    for (int y=0; y<ylen; ++y){
      int start = x*xstep + y*ystep;

      // printf("input:\n ");
      // for(int i=0;i<n;i++)printf(" %d ",in[start+skip*i]);
      // printf("\n");

      double v = 0;

      // handle leading edge of input (within radius of edge).
      int i = 0;
      for(i=0;i<kernel.radius;++i){
        v=0;
        for(int j=0,ji=start+skip*(i-kernel.radius);j<kernel.support;++j,ji+=skip){
          if(ji<start){   // out of bounds.
            v += kernel.data[j] * in[start];
          }
          else{        // in bounds.
            v += kernel.data[j] * in[ji];
          }
        }
        out[start+skip*i] = short(v);
      }

      // handle trailing edge of input.
      for(i=n-kernel.radius;i<n;++i){
        v=0;
        for(int j=0,ji=start+skip*(i-kernel.radius);j<kernel.support;++j,ji+=skip){
          if(ji>start+skip*(n-1)){
            v += kernel.data[j] * in[start+skip*(n-1)];
          }
          else{        // in bounds.
            v += kernel.data[j] * in[ji];
          }
        }
        out[start+skip*i] = short(v);
      }

      // convolve the center of the image.
      for(i=kernel.radius;i<n-kernel.radius;++i){
        v=0;
        for(int j=0,ji=start+skip*(i-kernel.radius);j<kernel.support;++j,ji+=skip){
          v += kernel.data[j] * in[ji];
          // printf(" + %.1f*%d", kernel.data[j], in[ji]);
        }
        // v = in[start+skip*i];
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

  conv2d(in, out, self.a1, self.a2, self.a3, self.w1, self.w2, self.w3, self.kernel);  // xy(z) axis.
  conv2d(out, in, self.a1, self.a3, self.a2, self.w1, self.w3, self.w2, self.kernel);  // x(y)z axis.
  conv2d(in, out, self.a2, self.a3, self.a1, self.w2, self.w3, self.w1, self.kernel);  // (x)yz axis.

  int from,to;

  // for(int z=self.a3-1;z>=0;z--){
  //   for(int y=self.a2-1;y>=0;y--){
  //     for(int x=self.a1-1;x>=0;x--){
  //       to = x*self.w1 + y*self.w2 + z*self.w3;
        
  //       if(  x<self.kernel.radius || y<self.kernel.radius || z<self.kernel.radius
  //         || x>=self.a1-self.kernel.radius || y>=self.a2-self.kernel.radius || z>=self.a3-self.kernel.radius){
  //           out[to] = 0;
  //         continue;
  //       }
        
  //       from = (x-self.kernel.radius)*self.w1 + (y-self.kernel.radius)*self.w2 + (z-self.kernel.radius)*self.w3;
  //       out[to] = out[from];
  //     }
  //   }
  // }
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
  // normalize so that the kernel sums to 1.0.
  double sum = 0;
  for(int i=0;i<k.support;++i)sum += k.data[i];
  for(int i=0;i<k.support;++i)k.data[i] /= sum;

  print_kernel(k);
  return k;
}
DiscreteKernel Filter::interpolation(){
  DiscreteKernel k;
  k.radius = 1;
  k.support = 3;
  k.data = new double[3];
  k.temp = new double[3];
  k.data[0] = 0.5;
  k.data[1] = 0;
  k.data[2] = 0.5;
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

// double Filter::comp_max_laplacian(short *in){
//   double max = 0;
//   double lap;
//   int xi, xm;
//   for(int x=boundary;x<self.a1;++x){
//     for(int y=boundary;y<self.a2;++y){
//       xi = x*self.w1 + y*self.w2 + 1*self.w3;
//       for(; xi<self.w4-self.w3; xi+=self.w3){
//         lap = 2.0*in[xi] - in[xi-self.w3] - in[xi+self.w3];
//         if(lap>max)max=lap;
//       }
//     }
//   }
//   for(int x=boundary;x<self.a1;++x){
//     for(int z=boundary;z<self.a3;++z){
//       xi = x*self.w1 + 1*self.w2           + z*self.w3;
//       xm = x*self.w1 + (self.a2-1)*self.w2 + z*self.w3;
//       for(; xi<xm; xi+=self.w2){
//         lap = 2.0*in[xi] - in[xi-self.w2] - in[xi+self.w2];
//         if(lap>max)max=lap;
//       }
//     }
//   }

//   for(int y=boundary;y<self.a2;++y){
//     for(int z=boundary;z<self.a3;++z){
//       xi = 1*self.w1 + y*self.w2           + z*self.w3;
//       xm = (self.a1-1)*self.w1 + y*self.w2 + z*self.w3;
//       for(; xi<xm; xi+=self.w1){
//         lap = 2.0*in[xi] - in[xi-self.w1] - in[xi+self.w1];
//         if(lap>max)max=lap;
//       }
//     }
//   }
//   return 3.0*max/2.0;
// }
void Filter::laplacian3d(int boundary){
  short *in  = self.buff[self.curr];
  short *out = self.buff[itempbuf()];

  // printf("lap %p -> %p\n",in,out);
  // printf("nb %d %p %p\n",self.nbuf, self.buff[0], self.buff[1]);

  // printf("dims %d %d %d %d\n",self.a0,self.a1,self.a2,self.a3);
  // printf("max %d\n",self.w4);

  // double max_laplacian = comp_max_laplacian(in);
  int xo;
  for(int z=0;z<self.a3; z++){
    for(int y=0; y<self.a2; y++){
      for(int x=0; x<self.a1; x++){
        xo = x*self.w1 + y*self.w2 + z*self.w3;

        if(x<1+boundary || y<1+boundary || z<1+boundary || x>=self.a1-1-boundary || y>=self.a2-1-boundary || z>=self.a3-1-boundary){
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

        // printf("%d %d -> %d %d\n",xi, in[xi], p1, in[p1]);

        double l1=0,l2=0,l3=0;

        l1 = in[xi] - 0.5*in[p1] - 0.5*in[p2];
        l2 = in[xi] - 0.5*in[p3] - 0.5*in[p4];
        l3 = in[xi] - 0.5*in[p5] - 0.5*in[p6];
        // double mx = fmax(l1,fmax(l2,l3));
        // double mn = fmin(l1,fmin(l2,l3));
        // printf("%f %f %f -> %f %f\n",l1,l2,l3,mx,mn);
        v = (l1+l2+l3)/3.0;
        // v *= 30000.0/max_laplacian;
        // v = fabs(in[p5]);
        if(v<0)v=0;
        out[xo] = short(v);
      }
    }
  }
  self.curr = itempbuf();
}

inline int median(int a, int b, int c){
  if(a<b){            // a b
    if(b<c)return b;  // a b c
    if(c<a)return a;  // c a b
    else   return c;  // a c b
  }else{              // b a
    if(a<c)return a;  // b a c
    if(c<b)return b;  // c b a
    else   return c;  // b c a
  }
}
void Filter::max1(){
  short *in  = self.buff[self.curr];
  short *out = self.buff[itempbuf()];

  int xi,xm;
  for(int x=0;x<self.a1;++x){
    for(int y=0;y<self.a2;++y){
      xi = x*self.w1 + y*self.w2 + 1*self.w3;
      for(; xi<self.w4-self.w3; xi+=self.w3){
        out[xi] = median(in[xi], in[xi-self.w3],in[xi+self.w3]);
        // out[xi] = max(in[xi],max(in[xi-self.w3],in[xi+self.w3]));
      }
    }
  }
  for(int x=0;x<self.a1;++x){
    for(int z=0;z<self.a3;++z){
      xi = x*self.w1 + 1*self.w2           + z*self.w3;
      xm = x*self.w1 + (self.a2-1)*self.w2 + z*self.w3;
      for(; xi<xm; xi+=self.w2){
        in[xi] = median(out[xi],out[xi-self.w2],out[xi+self.w2]);
        // in[xi] = max(out[xi],max(out[xi-self.w2],out[xi+self.w2]));
      }
    }
  }

  for(int y=0;y<self.a2;++y){
    for(int z=0;z<self.a3;++z){
      xi = 1*self.w1 + y*self.w2           + z*self.w3;
      xm = (self.a1-1)*self.w1 + y*self.w2 + z*self.w3;
      for(; xi<xm; xi+=self.w1){
        out[xi] = median(in[xi],in[xi-self.w1],in[xi+self.w1]);
        // out[xi] = max(in[xi],max(in[xi-self.w1],in[xi+self.w1]));
      }
    }
  }
  // for(int z=0;z<self.a3; z++){
  //   for(int y=0; y<self.a2; y++){
  //     for(int x=0; x<self.a1; x++){

  //       xi = x*self.w1 + y*self.w2 + z*self.w3;
        
  //       int neighbor[27];
  //       for(int i=0;i<27;i++){
  //         neighbor[i] = xi;
  //       }
  //       int nindex = 0;
  //       for(int xx=-1;xx<=1;++xx){
  //         for(int yy=-1;yy<=1;++yy){
  //           for(int zz=-1;zz<=1;++zz){
  //             int xii = xi + xx*self.w1 + yy*self.w2 + zz*self.w3;
  //             if(xii>=0 && xii < self.w4){
  //               neighbor[nindex] = xii;
  //               ++nindex;
  //             }
  //           }
  //         }
  //       }
  //       double v;

  //       v = max(
  //             max(
  //               max(
  //                 max(
  //                   max(in[neighbor[0]],in[neighbor[1]]),
  //                   max(in[neighbor[2]],in[neighbor[3]])),
  //                 max(
  //                   max(in[neighbor[4]],in[neighbor[5]]),
  //                   max(in[neighbor[6]],in[neighbor[7]]))),
  //               max(
  //                 max(
  //                   max(in[neighbor[8]],in[neighbor[9]]),
  //                   max(in[neighbor[10]],in[neighbor[11]])),
  //                 max(
  //                   max(in[neighbor[12]],in[neighbor[13]]),
  //                   max(in[neighbor[14]],in[neighbor[15]])))),
  //             max(
  //               max(
  //                 max(
  //                   max(in[neighbor[16]],in[neighbor[17]]),
  //                   max(in[neighbor[18]],in[neighbor[19]])),
  //                 max(
  //                   in[neighbor[20]],
  //                   max(in[neighbor[21]],in[neighbor[22]]))),
  //               max(
  //                 max(in[neighbor[23]],in[neighbor[24]]),
  //                 max(in[neighbor[25]],in[neighbor[26]]))));

  //       out[xi] = short(v);
  //     }
  //   }
  // }

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
    // int bucket = (int)(r*10.0);
    // if(bucket<0)bucket=0;
    // if(bucket>9)bucket=9;
    // buckets[bucket]++;
  }
  // printf("histogram:\n ");
  // for(int i=0;i<10;i++){
  //   printf("%4d, ",buckets[i]);
  // }
  // printf("\n");
  // printf("minmax: %.1f %.1f\n",min,min+max);
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
  return c;
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
  printf("found %d maxima.\n",maxima.size());
  return maxima;
}
void Filter::highlight(std::vector<glm::ivec3> points){
  // printf("maxima: %d\n",points.size());
  int max = 0;
  
  short *buff = self.buff[self.curr];

  for(int i=0;i<self.w4;i+=self.w1){
    if(buff[i]>max)max=buff[i];
    buff[i] = (buff[i]*3)/4;
  }
  for(int i=0;i<points.size();++i){
    glm::ivec3 v = points[i];
    int xi = v.x*self.w1 + v.y*self.w2 + v.z*self.w3;
    if(xi>=0 && xi<self.w4)buff[xi] = max;
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

struct Point{
  glm::ivec3 p;
  int   i;
};

std::vector<ScaleBlob*> Filter::find_blobs(){
  using namespace glm;

  std::vector<ivec3> maxima;
  // std::vector<glm::ivec3> maxima = find_maxima();
  // printf("maxima: %d\n",maxima.size());

  int *labelled = new int[self.w4];  // mark each point with a label indicated which cell it's part of.
  for(int i=0;i<self.w4;i++){        // initialize to -1.
    labelled[i] = -1;
  }
  short *data     = self.buff[self.curr];

  // breadcumb trail that we leave behind as we look for the max.
  // when we find the max, also update the maxes for each trailing
  // point that we also visited.
  const int max_trail_len = 10000;
  int trail[max_trail_len];
  int trail_length = 0; // length of the trail.

  int x=0,y=0,z=0;

  // hill-climb to determine labels.
  for(int z=0;z<self.a3; z++){
    if(z%50 == 0)printf("find_blobs progress %d/%d\n",z,self.a3);
    for(int y=0; y<self.a2; y++){
      for(int x=0; x<self.a1; x++){
        int i = x*self.w1 + y*self.w2 + z*self.w3;
      
        // starting position is this pixel.
        Point peak;
        peak.p = ivec3(x,y,z);
        peak.i = i;

        // printf("(%d %d %d %d)\n",peak.p.x, peak.p.y, peak.p.z, data[peak.i]);
        trail_length = 0;
        for(;;){
          // printf("  -> (%d %d %d %d) ",peak.p.x, peak.p.y, peak.p.z, data[peak.i]);
          int   maxi = INT_MIN;  // max neighbor index
          int   maxv = INT_MIN;  // max neighbor value
          ivec3 maxp = ivec3(0); // max neighbor coordinates

          if(labelled[peak.i] != -1){
            // we have reached a pixel that already has a label.
            // use this pixel's label as our own (if we keep 
            // climbing we'll reach the same point anyway).

            labelled[i] = labelled[peak.i];
            break;
          }

          // add to the trail.
          if(trail_length < max_trail_len){
            trail[trail_length] = peak.i;
            trail_length++;
          }


          // search for a higher neighbor.
          // do not change the order of traversal.
          // if there are ties, then the lowest index is chosen.
          for(int zz=max(peak.p.z-1,0);zz<=min(peak.p.z+1,self.a3-1); zz++){
            for(int yy=max(peak.p.y-1,0);yy<=min(peak.p.y+1,self.a2-1); yy++){
              for(int xx=max(peak.p.x-1,0);xx<=min(peak.p.x+1,self.a1-1); xx++){
                // printf("  %d %d %d\n",xx,yy,zz);
                int ii = xx*self.w1 + yy*self.w2 + zz*self.w3;
                int testv = data[ii];
                if(testv > maxv){
                  maxv = testv;
                  maxi = ii;
                  maxp = ivec3(xx,yy,zz);
                }
              }
            }
          }
          if(maxi == peak.i){
            // this is a peak. label it with the index of the maxima.
            labelled[i] = peak.i;
            break;
          }
          peak.p = maxp;
          peak.i = maxi;
        }

        // We have successfully labelled this pixel.
        // Now label all the points that we traversed while getting here.
        for(int j=0;j<trail_length;j++){
          labelled[trail[j]] = labelled[i];
        }
      } 
    }
  }

  // now, each pixel knows where its peak is. form a list of unique peaks.

  // short *output = self.buff[self.curr];
  // for(int i=0;i<self.w4;i++){
  //   output[i] = output[i]*3/4;
  // }
  // for(int i=0;i<self.w4;i++){
  //   output[labelled[i]] = 30000;
  // }

  // map labels to volume.
  std::unordered_map<int,int> labels_to_counts;
  for(int i=0;i<self.w4;i+=2){
    labels_to_counts[labelled[i]] = 0;
  }
  for(int i=0;i<self.w4;i+=2){
    labels_to_counts[labelled[i]] = labels_to_counts[labelled[i]] + 1;
  }

  // construct a mapping from index to blob.
  // also discard small blobs.
  std::unordered_map<int, ScaleBlob*> blobs;
  for ( auto it = labels_to_counts.begin(); it != labels_to_counts.end(); ++it ){
    int index = it->first;
    int count = it->second;
    if(count > 64){ // arbitrary minimum size, as an optimization.
      blobs[index] = new ScaleBlob();
    }
  }

  printf("compute blob statistics.\n");
  // now construct the scaleblobs with data in 2 passes.
  for(int pass=0;pass<2;++pass){
    for(int i=0;i<self.w4;i+=2){
      auto blob = blobs.find(labelled[i]);
      if(blob != blobs.end()){
        double x = (i/self.w1)%self.a1;
        double y = (i/self.w2)%self.a2;
        double z = (i/self.w3)%self.a3;
        blob->second->pass(glm::dvec3(x,y,z), double(data[i]));
      }
    }
    for (std::pair<int, ScaleBlob*> blob : blobs){
      blob.second->commit();
    }
  }

  printf("list blobs:\n");
  for (std::pair<int, ScaleBlob*> blob : blobs){
    blob.second->print();
  }

  std::vector<ivec3>       positions;
  std::vector<ScaleBlob*> output;

  for (std::pair<int, ScaleBlob*> blob : blobs){
    output.push_back(blob.second);
    positions.push_back(ivec3(blob.second->position));
  }

  // construct a sorted list of unique labels from this set.
  // std::vector<int> labels;
  // labels.assign( set.begin(), set.end() );
  // sort( labels.begin(), labels.end() );

  // construct a list of counts.
  // std::vector<int> counts(labels.size());

  // std::vector<ivec3>

  highlight(positions);
  printf("number of blobs: %d\n",output.size());
  delete[] labelled;
  printf("done.\n");
  return output;
}

void Filter::print(){

  int channel = 0;  
  int buckets[10];
  for(int i=0;i<10;++i)buckets[i] = 0;

  printf("axes %d %d %d %d\n",self.a0,self.a1,self.a2,self.a3);

  NrrdAxisInfo *a = self.a;
  short *data = self.buff[self.curr];

  double max = 0;
  double min = 30000;
  for(int i=channel;i<self.w4;i+=2){
    if(data[i] < min)min=data[i];
    if(data[i] > max)max=data[i];
  }
  max = max-min;
  for(int i=channel;i<self.w4;i+=2){
    double r = double(data[i]-min)/double(max);
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
}

void Filter::draw_blobs(std::vector<ScaleBlob*> blobs){
  for(int z=0;z<self.a3; z++){
    if(z%50 == 0)printf("find_blobs progress %d/%d\n",z,self.a3);
    for(int y=0; y<self.a2; y++){
      for(int x=0; x<self.a1; x++){
        int i = x*self.w1 + y*self.w2 + z*self.w3;
      }
    }
  }
}