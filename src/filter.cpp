#include "filter.h"
#include "HuangQS.h"
#include <vector>
#include <queue> 
#include <unordered_set>
#include <unordered_map>
#include <unistd.h>
#include <climits>
#include <cmath>
#include <limits>
#include "util.h"
#define pi (3.14159265358979323846264)
#define gaus(x,stdv) (exp(-((x)*(x))/(2*(stdv)*(stdv)))/((stdv)*sqrt(2*pi)))
#define gausd2(x,sig)(exp(-x*x/(2.0*sig*sig))*(x*x-sig*sig) /       \
     (sig*sig*sig*sig*sig*2.50662827463100050241))


DiscreteKernel::DiscreteKernel(){
  data = 0;
  temp = 0;
}
DiscreteKernel::~DiscreteKernel(){
  // if(data) delete[] data;
  // if(temp) delete[] temp;
}
void DiscreteKernel::destroy(){
  if(data) delete[] data;
  if(temp) delete[] temp;
}
struct thread_conv2d_info_shared{
  float *in;
  float *out;
  int xbegin;
  int xlen;
  int ylen;
  int zlen;
  int xstep;
  int ystep;
  int zstep;
  DiscreteKernel kernel;
};

void ArFilter::conv2d(float *in, float *out, int xlen, int ylen, int zlen, int xstep, int ystep, int zstep, DiscreteKernel kernel){
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
          else if(ji<=start+skip*(n-1)){        // in bounds.
            v += kernel.data[j] * in[ji];
          }
        }
        // v=kernel.data[kernel.radius]*2.f;
        out[start+skip*i] = float(v);
      }

      // handle trailing edge of input.
      for(i=n-kernel.radius;i<n;++i){
        v=0;
        for(int j=0,ji=start+skip*(i-kernel.radius);j<kernel.support;++j,ji+=skip){
          if(ji>start+skip*(n-1)){
            v += kernel.data[j] * in[start+skip*(n-1)];
          }
          else if(ji>=0){        // in bounds.
            v += kernel.data[j] * in[ji];
          }
        }
        // v=kernel.data[kernel.radius];
        out[start+skip*i] = float(v);
      }

      // convolve the center of the image.
      for(i=kernel.radius;i<n-kernel.radius;++i){
        v=0;
        for(int j=0,ji=start+skip*(i-kernel.radius);j<kernel.support;++j,ji+=skip){
          v += kernel.data[j] * in[ji];
          // printf(" + %.1f*%d", kernel.data[j], in[ji]);
        }
        // v = in[start+skip*i];
        out[start+skip*i] = float(v);
        // printf(" = %.1f\n",v);
      }

      // printf("output:\n ");
      // for(int i=0;i<n;i++)printf(" %d ",out[start+skip*i]);
      // printf("\n");

      // exit(0);
    }
  }
}
void ArFilter::filter(){
  // printf("curr=%d\n", self.curr);
  float *in  = self.buff[self.curr];
  float *out = self.buff[itempbuf()];

  // printf("conv: %p -> %p\n", in, out);
  conv2d(in, out, self.a0, self.a1, self.a2, self.w0, self.w1, self.w2, self.kernel);  // xy(z) axis.
  // printf("conv: %p -> %p\n", out, in);
  conv2d(out, in, self.a0, self.a2, self.a1, self.w0, self.w2, self.w1, self.kernel);  // x(y)z axis.
  // printf("conv: %p -> %p\n", in, out);
  conv2d(in, out, self.a1, self.a2, self.a0, self.w1,  self.w2, self.w0, self.kernel);  // (x)yz axis.

  int from,to;

  // for(int z=self.a2-1;z>=0;z--){
  //   for(int y=self.a1-1;y>=0;y--){
  //     for(int x=self.a0-1;x>=0;x--){
  //       to = x*self.w0 + y*self.w1 + z*self.w2;
        
  //       if(  x<self.kernel.radius || y<self.kernel.radius || z<self.kernel.radius
  //         || x>=self.a0-self.kernel.radius || y>=self.a1-self.kernel.radius || z>=self.a2-self.kernel.radius){
  //           out[to] = 0;
  //         continue;
  //       }
        
  //       from = (x-self.kernel.radius)*self.w0 + (y-self.kernel.radius)*self.w1 + (z-self.kernel.radius)*self.w2;
  //       out[to] = out[from];
  //     }
  //   }
  // }
  self.curr = itempbuf();
}
DiscreteKernel ArFilter::gaussian(double sigma, int radius, int d){
  // printf("create gaussian kernel: %.2f, %d\n",sigma, radius);
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

  // print_kernel(k);
  return k;
}
DiscreteKernel ArFilter::interpolation(){
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
void ArFilter::set_kernel(DiscreteKernel k){
  this->self.kernel = k;
}
void ArFilter::print_kernel(DiscreteKernel k){
  printf("kernel...\n  ");
  for(int i=0;i<k.support;++i){
    printf("%2.3f ",k.data[i]);
  }
  printf("\n/kernel...\n");
}
// struct filter_info{
//   int len;
//   float *in;
//   float *out;
// };
// static filter_info get_info(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info fi;
//   NrrdAxisInfo *a = nin->axis;
//   fi.len = a[0].size * a[1].size * a[2].size *a[3].size;
//   fi.in = (float*)nin->data;
//   fi.out  = (float*)nout->data;
//   return fi;
// }
// void ArFilter::positive(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info f = get_info(nin, nout, channel);
//   for(int i=channel;i<f.len;i+=2){
//     if(f.in[i]>0)f.out[i]=f.in[i];
//     else f.out[i] = 0;
//   }
// }
// void ArFilter::negative(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info f = get_info(nin, nout, channel);
//   for(int i=channel;i<f.len;i+=2){
//     if(f.in[i]<0)f.out[i]= f.in[i];
//     else f.out[i] = 0;
//   }
// }
// void ArFilter::binary(Nrrd *nin, Nrrd *nout, int channel){
//   filter_info f = get_info(nin, nout, channel);
//   for(int i=channel;i<f.len;i+=2){
//     if(f.in[i]>0)f.out[i]=100;
//     else f.out[i] = 0;
//   }
// }

// double ArFilter::comp_max_laplacian(float *in){
//   double max = 0;
//   double lap;
//   int xi, xm;
//   for(int x=boundary;x<self.a0;++x){
//     for(int y=boundary;y<self.a1;++y){
//       xi = x*self.w0 + y*self.w1 + 1*self.w2;
//       for(; xi<self.w3-self.w2; xi+=self.w2){
//         lap = 2.0*in[xi] - in[xi-self.w2] - in[xi+self.w2];
//         if(lap>max)max=lap;
//       }
//     }
//   }
//   for(int x=boundary;x<self.a0;++x){
//     for(int z=boundary;z<self.a2;++z){
//       xi = x*self.w0 + 1*self.w1           + z*self.w2;
//       xm = x*self.w0 + (self.a1-1)*self.w1 + z*self.w2;
//       for(; xi<xm; xi+=self.w1){
//         lap = 2.0*in[xi] - in[xi-self.w1] - in[xi+self.w1];
//         if(lap>max)max=lap;
//       }
//     }
//   }

//   for(int y=boundary;y<self.a1;++y){
//     for(int z=boundary;z<self.a2;++z){
//       xi = 1*self.w0 + y*self.w1           + z*self.w2;
//       xm = (self.a0-1)*self.w0 + y*self.w1 + z*self.w2;
//       for(; xi<xm; xi+=self.w0){
//         lap = 2.0*in[xi] - in[xi-self.w0] - in[xi+self.w0];
//         if(lap>max)max=lap;
//       }
//     }
//   }
//   return 3.0*max/2.0;
// }
void ArFilter::laplacianmasked(float scale){
  float *raw = self.buff[self.curr];
  int iraw = self.curr;
  laplacian3d();
  float *lap = self.buff[self.curr];
  for(int i=0;i<self.w3;i++){
    if(lap[i] == 0)raw[i] = 0;
  }
  self.curr = iraw;

}
void ArFilter::laplacian3d(int boundary){
  float *in  = self.buff[self.curr];
  float *out = self.buff[itempbuf()];

  // printf("lap %p -> %p\n",in,out);
  // printf("nb %d %p %p\n",self.nbuf, self.buff[0], self.buff[1]);

  // printf("dims %d %d %d %d\n",self.a0,self.a0,self.a1,self.a2);
  // printf("max %d\n",self.w3);

  // double max_laplacian = comp_max_laplacian(in);
  int xo;
  for(int z=0;z<self.a2; z++){
    for(int y=0; y<self.a1; y++){
      for(int x=0; x<self.a0; x++){
        xo = x*self.w0 + y*self.w1 + z*self.w2;

        if(x<1+boundary || y<1+boundary || z<1+boundary || x>=self.a0-1-boundary || y>=self.a1-1-boundary || z>=self.a2-1-boundary){
          out[xo]=0;
          continue;
        }
        
        int p1=0,p2=0,p3=0,
            p4=0,p5=0,p6=0;
        double v;

        int xi = xo;
        p1 = xi-self.w0;
        p2 = xi+self.w0;
        p3 = xi-self.w1;
        p4 = xi+self.w1;
        p5 = xi-self.w2;
        p6 = xi+self.w2;

        // printf("%d %d -> %d %d\n",xi, in[xi], p1, in[p1]);

        double l1=0,l2=0,l3=0;

        l1 = 2*in[xi] - in[p1] - in[p2];
        l2 = 2*in[xi] - in[p3] - in[p4];
        l3 = 2*in[xi] - in[p5] - in[p6];
        // double mx = fmax(l1,fmax(l2,l3));
        // double mn = fmin(l1,fmin(l2,l3));
        // printf("%f %f %f -> %f %f\n",l1,l2,l3,mx,mn);
        v = (l1+l2+l3);
        // v *= 30000.0/max_laplacian;
        // v = fabs(in[p5]);
        if(v<0)v=0;
        out[xo] = float(v);
      }
    }
  }
  self.curr = itempbuf();
}

inline float fmedian(float a, float b, float c){
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
void ArFilter::max1(){
  float *in  = self.buff[self.curr];
  float *out = self.buff[itempbuf()];

  // printf("max1 %p -> %p\n", in, out);
  int xi,xm;

  // for(int x=0;x<self.a0;++x){
  //   for(int y=0;y<self.a1;++y){
  //     xi = x*self.w0 + y*self.w1 + 1*self.w2;
  //     for(; xi<self.w3-self.w2; xi+=self.w2){
  //       // out[xi] = 1.f;
  //       // out[xi+1] = in[xi];
  //       // out[xi] = in[xi];
  //       // out[xi] = in[xi+self.w2];
  //       out[xi] = fmedian(in[xi], in[xi-self.w2],in[xi+self.w2]);
  //       // out[xi] = max(in[xi],max(in[xi-self.w2],in[xi+self.w2]));
  //     }
  //   }
  // }
  // for(int x=0;x<self.a0;++x){
  //   for(int z=0;z<self.a2;++z){
  //     xi = x*self.w0 + 1*self.w1           + z*self.w2;
  //     xm = x*self.w0 + (self.a1-1)*self.w1 + z*self.w2;
  //     for(; xi<xm; xi+=self.w1){
  //       // in[xi] = out[xi];
  //       in[xi] = fmedian(out[xi],out[xi-self.w1],out[xi+self.w1]);
  //       // in[xi] = max(out[xi],max(out[xi-self.w1],out[xi+self.w1]));
  //     }
  //   }
  // }

  // for(int y=0;y<self.a1;++y){
  //   for(int z=0;z<self.a2;++z){
  //     xi = 1*self.w0 + y*self.w1           + z*self.w2;
  //     xm = (self.a0-1)*self.w0 + y*self.w1 + z*self.w2;
  //     for(; xi<xm; xi+=self.w0){
  //       // out[xi] = in[xi];
  //       out[xi] = fmedian(in[xi],in[xi-self.w0],in[xi+self.w0]);
  //       // out[xi] = max(in[xi],max(in[xi-self.w0],in[xi+self.w0]));
  //     }
  //   }
  // }
  for(int z=0;z<self.a2; z++){
    for(int y=0; y<self.a1; y++){
      for(int x=0; x<self.a0; x++){

        xi = x*self.w0 + y*self.w1 + z*self.w2;
        
        int neighbor[27];
        for(int i=0;i<27;i++){
          neighbor[i] = xi;
        }
        int nindex = 0;
        for(int xx=-1;xx<=1;++xx){
          for(int yy=-1;yy<=1;++yy){
            for(int zz=-1;zz<=1;++zz){
              int xii = xi + xx*self.w0 + yy*self.w1 + zz*self.w2;
              if(xii>=0 && xii < self.w3){
                neighbor[nindex] = xii;
                ++nindex;
              }
            }
          }
        }
        double v;
        v = fmedian(
          fmedian(
            fmedian(in[neighbor[0]], in[neighbor[1]], in[neighbor[2]]),
            fmedian(in[neighbor[3]], in[neighbor[4]], in[neighbor[5]]),
            fmedian(in[neighbor[6]], in[neighbor[7]], in[neighbor[8]])),
          fmedian(
            fmedian(in[neighbor[9]], in[neighbor[10]], in[neighbor[11]]),
            fmedian(in[neighbor[12]], in[neighbor[13]], in[neighbor[14]]),
            fmedian(in[neighbor[15]], in[neighbor[16]], in[neighbor[17]])),
          fmedian(
            fmedian(in[neighbor[18]], in[neighbor[19]], in[neighbor[20]]),
            fmedian(in[neighbor[21]], in[neighbor[22]], in[neighbor[23]]),
            fmedian(in[neighbor[24]], in[neighbor[25]], in[neighbor[26]])));

        // v = max(
        //       max(
        //         max(
        //           max(
        //             max(in[neighbor[0]],in[neighbor[1]]),
        //             max(in[neighbor[2]],in[neighbor[3]])),
        //           max(
        //             max(in[neighbor[4]],in[neighbor[5]]),
        //             max(in[neighbor[6]],in[neighbor[7]]))),
        //         max(
        //           max(
        //             max(in[neighbor[8]],in[neighbor[9]]),
        //             max(in[neighbor[10]],in[neighbor[11]])),
        //           max(
        //             max(in[neighbor[12]],in[neighbor[13]]),
        //             max(in[neighbor[14]],in[neighbor[15]])))),
        //       max(
        //         max(
        //           max(
        //             max(in[neighbor[16]],in[neighbor[17]]),
        //             max(in[neighbor[18]],in[neighbor[19]])),
        //           max(
        //             in[neighbor[20]],
        //             max(in[neighbor[21]],in[neighbor[22]]))),
        //         max(
        //           max(in[neighbor[23]],in[neighbor[24]]),
        //           max(in[neighbor[25]],in[neighbor[26]]))));

        out[xi] = float(v);
      }
    }
  }

  self.curr = itempbuf();
  // printf("max1 -> %p\n", self.buff[self.curr]);
}

void ArFilter::threshold(float min, float max){
  float *data = self.buff[self.curr];
  for(int i=0;i<self.w3;i++){
    if(data[i]<min)data[i]=0;
    if(data[i]>max)data[i]=max;
  }
}
void ArFilter::normalize(double power){

  NrrdAxisInfo *a = self.a;
  float *data = self.buff[self.curr];
  float *out  = self.buff[itempbuf()];

  float max = 0;
  float min = std::numeric_limits<float>::infinity();
  for(int i=0;i<self.w3;i+=self.w0){
    if(data[i] < min)min=data[i];
    if(data[i] > max)max=data[i];
  }

  max = max-min;
  for(int i=0;i<self.w3;i+=self.w0){
    float r = (data[i]-min)/(max);
    if(power == 1)out[i] = r;
    else out[i] = pow(r,power);
  }
  self.curr = itempbuf();
}
void ArFilter::scale(float s){
  float *data = self.buff[self.curr];
  for(int i=0;i<self.w3;i+=self.w0){
    data[i] *= s;
  }
}
// void ArFilter::median1(){
//   float *in = self.buff[self.curr];
//   float *out = self.buff[itempbuf()];
//   NrrdAxisInfo *a = self.a;

//   int a1 = a[1].size;
//   int a2 = a[2].size;
//   int a3 = a[3].size;

//   float *bufin  = new float[a1*a2*a3];
//   float *bufout = new float[a1*a2*a3];

//   // float *bufin  = (float*) malloc(sizeof(float)*((a->size[1])*(a->size[2])*(a->size[3])));
//   // float *bufout = (float*) malloc(sizeof(float)*((a->size[1])*(a->size[2])*(a->size[3])));
//   int dims[3]  = {a1,a2,a3};
//   int fmin[3]  = {0,0,0};
//   int fsiz[3] = {a1,a2,a3};
//   int fmax[3] = {a1,a2,a3};
//   int l = a1*a2*a3;
//   for(int i=0;i<l;i++){
//     bufin[i] = in[i*2];
//   }
//   median_filter_3D<1>(bufin, dims, bufout, fmin, fsiz, fmax);
//   for(int i=0;i<l;i++){
//     out[i*2] = bufout[i];
//   }
//   free(bufin);
//   free(bufout);
//   self.curr = itempbuf();
// }

void ArFilter::capture(Nrrd *nin){
  if(nin->axis[0].size == self.a0 &&
     nin->axis[1].size == self.a1 &&
     nin->axis[2].size == self.a2){
    memcpy(self.buff[self.curr], nin->data, self.w3 * sizeof(float));
  }
}
void ArFilter::init(Nrrd *nin){
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
  // self.a2=self.a[3].size;

  // self.w0=1;     // offset to crawl channel
  self.w0=1;               // offset to crawl x
  self.w1=self.a0*self.w0; // offset to crawl y
  self.w2=self.a1*self.w1; // offset to crawl z
  self.w3=self.a2*self.w2; // length of entire dataset.

  self.nbuf = 2;
  self.nrrd = new Nrrd* [self.nbuf];
  self.buff = new float*[self.nbuf];
  self.nrrd[0] = nin;
  self.buff[0] = (float*)nin->data;

  for(int i=1;i<self.nbuf;++i){
    self.nrrd[i] = nrrdNew();
    nrrdCopy(self.nrrd[i],nin);
    self.buff[i] = (float*)(self.nrrd[i]->data);
    // printf("self.buff[i] = %p %p\n", nin->data, self.nrrd[i]->data);
  }
  // printf("filter::init %p %p\n", self.nrrd[0], self.nrrd[1]);
  // printf("filter::init %p %p\n", self.buff[0], self.buff[1]);
  self.curr = 0;
  self.alive = true;
}
Nrrd* ArFilter::commit(Nrrd *nout){
  if(!nout)nout = self.nrrd[0];
  if(0 != nout){
    // printf("memcpy %p %p\n", nout->data, self.nrrd[self.curr]);
    // memset(self.nrrd[self.curr], 0, sizeof(float)*self.w3);
    // printf("commit %p -> %p.\n", self.buff[self.curr], nout);
    memcpy(nout->data, self.buff[self.curr], sizeof(float)*self.w3);
    // printf("done.\n");
    // exit(0);
  }
  return nout;
}
void ArFilter::destroy(){
  // printf("Destroy.\n");
  for(int i=1;i<self.nbuf;++i){
    nrrdNuke(self.nrrd[i]);
  }
  if(self.kernel.data)delete[] self.kernel.data;
  if(self.kernel.temp)delete[] self.kernel.temp;
  init(0);
}
int ArFilter::itempbuf(int c){
  c++;
  if(c>=self.nbuf){
    return 0;
  }
  return c;
}
int ArFilter::itempbuf(){
  return itempbuf(self.curr);
}
ArFilter::ArFilter(){
  init(0);
}

ivec3 ArFilter::hill_climb(ivec3 p){
  // printf("%d %d %d; ", p.x, p.y, p.z);
  float *data = self.buff[self.curr];

  if(p.x<1)p.x = 1;
  if(p.y<1)p.y = 1;
  if(p.z<1)p.z = 1;
  if(p.x>self.a0-2)p.x = self.a0-2;
  if(p.y>self.a1-2)p.y = self.a1-2;
  if(p.z>self.a2-2)p.z = self.a2-2;

  float maxv = -1;
  ivec3 maxp;

  int besti = -1;
  int maxi = -1;
  int ii2 = -1;

  bool climbing = true;

  // printf("climbing...");

  while(climbing){
    climbing = false;
    for(int     zz=p.z-1; zz<=p.z+1; zz++){
      for(int   yy=p.y-1; yy<=p.y+1; yy++){
        for(int xx=p.x-1; xx<=p.x+1; xx++){
          // printf("  %d %d %d\n",xx,yy,zz);
          int ii = xx*self.w0 + yy*self.w1 + zz*self.w2;
          float testv = data[ii];
          if(testv > maxv){
            maxv = testv;
            maxp = ivec3(xx,yy,zz);
            // printf("%d %d %d; ", xx, yy, zz);
            climbing = true;
          }
        }
      }
    }
  }
  // printf("\n");
  return maxp;
}
void ArFilter::lapofgaussian(float sigma){
  DiscreteKernel kernel = gaussian(sigma, int(sigma*4));
  set_kernel(kernel);
  filter();
  laplacian3d();
  normalize();
  kernel.destroy();
}
std::vector<glm::ivec3> ArFilter::find_maxima(){
  std::vector<glm::ivec3> maxima;

  float *data = self.buff[self.curr];

  for(int z=0;z<self.a2; z++){
    for(int y=0; y<self.a1; y++){
      for(int x=0; x<self.a0; x++){

        if(x<1 || y<1 || z<1 || x>=self.a0-1 || y>=self.a1-1 || z>=self.a2-1){
          continue;
        }

        int xi = x*self.w0 + y*self.w1 + z*self.w2;
        float p1=0,p2=0,p3=0,
              p4=0,p5=0,p6=0, v;


        v  = data[xi];
        p1 = data[xi-self.w0];
        p2 = data[xi+self.w0];
        p3 = data[xi-self.w1];
        p4 = data[xi+self.w1];
        p5 = data[xi-self.w2];
        p6 = data[xi+self.w2];

        // if(v>0)printf("v: %.4f\n",v);

        if(v>=p1 && v>=p2 && v>=p3 && v>=p4 && v>=p5 && v>=p6 && v != 0 ){
          maxima.push_back(glm::ivec3(x,y,z));
        }
      }
    }
  }
  printf("found %lu maxima.\n",maxima.size());
  return maxima;
}
void ArFilter::highlight(std::vector<glm::ivec3> points){

  float *buff = self.buff[self.curr];
  for(int i=0;i<self.w3;i+=self.w0){
    buff[i] = buff[i]*0.5f;
  }
  for(int i=0;i<points.size();++i){
    glm::ivec3 v = points[i];
    int xi = v.x*self.w0 + v.y*self.w1 + v.z*self.w2;
    if(xi>=0 && xi<self.w3)buff[xi] = 1.f;
  }
}

// given an input image, find all blobs.
// a blob is a local maximum surrounded by
// its hinterland. ie.:
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

std::vector<ScaleBlob*> ArFilter::find_blobs(){
  using namespace glm;

  std::vector<ivec3> maxima;
  // std::vector<glm::ivec3> maxima = find_maxima();
  // printf("maxima: %d\n",maxima.size());

  int *labelled = new int[self.w3];  // mark each point with a label indicated which cell it's part of.
  for(int i=0;i<self.w3;i++){        // initialize to -1.
    labelled[i] = -1;
  }
  float *data     = self.buff[self.curr];

  // breadcumb trail that we leave behind as we look for the max.
  // when we find the max, also update the maxes for each trailing
  // point that we also visited.
  const int max_trail_len = 10000;
  int trail[max_trail_len];
  int trail_length = 0; // length of the trail.

  int x=0,y=0,z=0;

  // hill-climb to determine labels.
  for(int z=0;z<self.a2; z++){
    // if(z%50 == 0)printf("find_blobs progress %d/%d\n",z,self.a2);
    for(int y=0; y<self.a1; y++){
      for(int x=0; x<self.a0; x++){
        int i = x*self.w0 + y*self.w1 + z*self.w2;
        // printf("xy %d %d\n",x,y);
        // starting position is this pixel.
        Point peak;
        peak.p = ivec3(x,y,z);
        peak.i = i;
        // printf("%d %d\n",i, peak.i);

        // printf("(%d %d %d %d)\n",peak.p.x, peak.p.y, peak.p.z, data[peak.i]);
        trail_length = 0;
        for(;;){
          // printf("%d %d\n",i, peak.i);
          // printf("  -> (%d %d %d %d) ",peak.p.x, peak.p.y, peak.p.z, data[peak.i]);
          int   maxi = 0;  // max neighbor index
          float maxv = -1.f;      // max neighbor value
          ivec3 maxp = ivec3(0); // max neighbor coordinates
          // printf("3\n");

          // printf("labelled[%d]\n",peak.i);
          // printf("i=%d %d from %d %d %d\n", i, peak.i, x, y, z);
          if(labelled[peak.i] != -1){
            // we have reached a pixel that already has a label.
            // use this pixel's label as our own (if we keep 
            // climbing we'll reach the same point anyway).
            // printf("labelled[%i]\n",i);
            labelled[i] = labelled[peak.i];
            // printf("4\n");
            break;
          }
          // printf("5\n");

          // add to the trail.
          if(trail_length < max_trail_len){
            trail[trail_length] = peak.i;
            trail_length++;
          }

          // printf("0\n");

          // search for a higher neighbor.
          // do not change the order of traversal.
          // if there are ties, then the lowest index is chosen.
          for(int zz=max(peak.p.z-1,0);zz<=min(peak.p.z+1,self.a2-1); zz++){
            for(int yy=max(peak.p.y-1,0);yy<=min(peak.p.y+1,self.a1-1); yy++){
              for(int xx=max(peak.p.x-1,0);xx<=min(peak.p.x+1,self.a0-1); xx++){
                // printf("  %d %d %d\n",xx,yy,zz);
                int      ii = xx*self.w0 + yy*self.w1 + zz*self.w2;
                float testv = data[ii];
                if(testv > maxv){
                  maxv = testv;
                  maxi = ii;
                  maxp = ivec3(xx,yy,zz);
                }
              }
            }
          }
          // printf("1\n");
          if(maxi == peak.i){
            // this is a peak. label it with the index of the maxima.
            // printf("peak at %d = \n",peak.i,peak.i);
            labelled[i] = peak.i;
            break;
          }
          peak.p = maxp;
          peak.i = maxi;
        }
        // printf("2\n");

        // We have successfully labelled this pixel.
        // Now label all the points that we traversed while getting here.
        for(int j=0;j<trail_length;j++){
          labelled[trail[j]] = labelled[i];
        }
      } 
    }
  }

  // now, each pixel knows where its peak is. form a list of unique peaks.

  // float *output = self.buff[self.curr];
  // for(int i=0;i<self.w3;i++){
  //   output[i] = output[i]*3/4;
  // }
  // for(int i=0;i<self.w3;i++){
  //   output[labelled[i]] = 30000;
  // }

  // form a map, label -> how many voxels are in this blob.
  std::unordered_map<int,int> labels_to_counts;
  for(int i=0;i<self.w3;i+=2){            // form a list of unique labels.
    labels_to_counts[labelled[i]] = 0;
  }
  for(int i=0;i<self.w3;i+=2){            // count how many pixels have each label.
    labels_to_counts[labelled[i]] = labels_to_counts[labelled[i]] + 1;
  }

  // construct a mapping from index to blob.
  // also discard small blobs.
  std::unordered_map<int, ScaleBlob*> blobs;
  for ( auto it = labels_to_counts.begin(); it != labels_to_counts.end(); ++it ){
    int index = it->first;
    int count = it->second;
    if(count > 125){                       // impose (arbitrary) minimum blob size, as an optimization.
      blobs[index] = new ScaleBlob();
      float x = (index/self.w0)%self.a0;      // tell the blob where its center is.
      float y = (index/self.w1)%self.a1;
      float z = (index/self.w2)%self.a2;
      blobs[index]->mode = vec3(x,y,z);
      blobs[index]->peakvalue = data[index];  // tell the blob the value of its center
    }
  }

  // printf("compute blob statistics.\n");
  // now construct the scaleblobs with data in 2 passes.
  for(int pass=0;pass<2;++pass){
    for(int i=0;i<self.w3;i+=2){
      auto blob = blobs.find(labelled[i]);
      if(blob != blobs.end()){
        float x = (i/self.w0)%self.a0;
        float y = (i/self.w1)%self.a1;
        float z = (i/self.w2)%self.a2;
        blob->second->pass(glm::vec3(x,y,z), (data[i]));
      }
    }
    for (std::pair<int, ScaleBlob*> blob : blobs){
      blob.second->commit();
    }
  }

  // printf("list blobs:\n");
  // for (std::pair<int, ScaleBlob*> blob : blobs){
  //   blob.second->print();
  // }

  // std::vector<ivec3>       positions;
  std::vector<ScaleBlob*> output;

  for (std::pair<int, ScaleBlob*> blob : blobs){
    // another optimization: only add blobs with volume > 4.
    if(blob.second->detCov > 4.f){
      output.push_back(blob.second);
    }
    // positions.push_back(ivec3(blob.second->position));
  }

  // construct a sorted list of unique labels from this set.
  // std::vector<int> labels;
  // labels.assign( set.begin(), set.end() );
  // sort( labels.begin(), labels.end() );

  // construct a list of counts.
  // std::vector<int> counts(labels.size());

  // std::vector<ivec3>
  // normalize(0.1);
  // highlight(positions);
  printf("%lu blobs.\n",output.size());
  delete[] labelled;
  // printf("done.\n");
  return output;
}

void ArFilter::print(){

  int channel = 0;  
  int buckets[10];
  for(int i=0;i<10;++i)buckets[i] = 0;

  printf("axes %d %d %d %d\n",self.a0,self.a0,self.a1,self.a2);

  NrrdAxisInfo *a = self.a;
  float *data = self.buff[self.curr];

  double max = 0;
  double min = std::numeric_limits<double>::infinity();
  for(int i=channel;i<self.w3;i+=2){
    if(data[i] < min)min=data[i];
    if(data[i] > max)max=data[i];
  }
  max = max-min;
  for(int i=channel;i<self.w3;i+=2){
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

void ArFilter::clear(){
  float *data = self.buff[self.curr];
  for(int i=0;i<self.w3;i++)data[i] = 0;
}
void ArFilter::rasterlineadd(vec3 a, vec3 b, float va, float vb){
  float *data = self.buff[self.curr];
  int i;

  vec3 v = b-a;
  float len = length(b-a);
  vec3 step = v/len;

  for(int j=0;j<len;++j){
    a += step;
    i = int(a.x)*self.w0 + int(a.y)*self.w1 + int(a.z)*self.w2;
    // printf(". %.1f %.1f %.1f\n", a.x, a.y, a.z);
    data[i] = va + (float(j)/len)*(vb-va);
  }
}
void ArFilter::color_blobs(std::vector<ScaleBlob*> blobs, float color){
  using namespace glm;
  float *data = self.buff[self.curr];

  // iterate through blobs.
  for(auto blob = blobs.begin(); blob != blobs.end(); ++blob){
    ScaleBlob *sb = *blob;
    float minx = sb->min.x;
    float miny = sb->min.y;
    float minz = sb->min.z;
    float maxx = sb->max.x;
    float maxy = sb->max.y;
    float maxz = sb->max.z;

    // iterate through pixels for each blob.
    for(float x=minx; x<=maxx; ++x){
      for(float y=miny; y<=maxy; ++y){
        for(float z=minz; z<=maxz; ++z){
          int i = int(x)*self.w0 + int(y)*self.w1 + int(z)*self.w2;
          float v = sb->cellpdf(vec3(x,y,z));
          if(std::isfinite(v) && v > 0.0001f){
            float orig = data[i] - 2.f*int(data[i]/2.f);
            data[i] = color + max(orig,v);
          }
        }
      }
    }
  }
}
struct thread_drawblobs_info{
  ArFilter *filter;
  std::vector<ScaleBlob*> *blobs;
  float *data;
  float *lock;
  int blobmin;          // render window min value.
  int blobmax;          // render window max value.
  const char *mode;     // either 'q'=quick or 'g'=gaussian
                        // mode[2] either '+' or 'm'
};

static void* t_draw_blobs(void* vinfo){
  thread_drawblobs_info *info = (thread_drawblobs_info*)vinfo;
  float *data = info->data;
  float *lock = info->lock;
  int blobmin = info->blobmin;
  int blobmax = info->blobmax;
  const char *mode   = info->mode;
  ArFilter *filter = info->filter;
  std::vector<ScaleBlob*> &blobs = *info->blobs;

  // printf("t_draw_blobs %d - %d\n", blobmin, blobmax);
  for(int bi=blobmin; bi<blobmax; ++bi){
    ScaleBlob *sb = blobs[bi];
            // printf("dot");
    if(sb->n < 3)continue;
    int minx = sb->min.x;
    int miny = sb->min.y;
    int minz = sb->min.z;
    int maxx = sb->max.x;
    int maxy = sb->max.y;
    int maxz = sb->max.z;
    if(minx < 0)minx = 0;
    if(miny < 0)miny = 0;
    if(minz < 0)minz = 0;
    if(maxx >= filter->self.a0)maxx = filter->self.a0;
    if(maxy >= filter->self.a1)maxy = filter->self.a1;
    if(maxz >= filter->self.a2)maxz = filter->self.a2;
    float v;
    // printf("minmax %d %d %d %d %d %d\n",minx, maxx, miny, maxy, minz, maxz);
    for(int x=minx; x<=maxx; ++x){
      for(int y=miny; y<=maxy; ++y){
        for(int z=minz; z<=maxz; ++z){
          int i = (x)*filter->self.w0 + (y)*filter->self.w1 + (z)*filter->self.w2;
          if(i<0)continue;
          if(i>=filter->self.w3)continue;
          // float 
          if(mode[0] == 'g'){
            v = sb->pdf(vec3(x,y,z));
          }
          else if(mode[0]=='q'){
            v = sb->cellpdf(vec3(x,y,z));
          }
          else if(mode[0]=='e'){
            v = sb->cellerf(vec3(x,y,z));
          }
          else if(mode[0]=='.'){
            v = sb->celldot(vec3(x,y,z));
          }
          else if(mode[0]=='e'){
            v = sb->ellipsepdf(vec3(x,y,z));
          }
          else if(mode[0]=='m'){
            v = sb->generalized_multivariate_gaussian_pdf(vec3(x,y,z));
          }
          // float v = 0.5f;
          // if(std::isfinite(v)){
          if(mode[1] == '+'){
            data[i] = (data[i] + v);
            if(data[i]>1)data[i] = 1;
          }
          else data[i] = max(data[i],v);
            // printf("(%.2f %.2f %.2f) -> %.f", x,y,z,v);
          // }
        }
      }
    }
  }
}

void ArFilter::draw_blobs(std::vector<ScaleBlob*> blobs, const char *mode){
  // printf("draw blobs.");
  using namespace glm;

  const int nthreads = 12;
  float *lock  = self.buff[itempbuf()];
  float *data  = self.buff[self.curr];

  pthread_t threads[nthreads];
  thread_drawblobs_info info[nthreads];


  for(int i=0;i<nthreads;i++){
    info[i].blobs = &blobs;
    info[i].lock = lock;
    info[i].data = data;
    info[i].filter = this;

    info[i].blobmin = (i*blobs.size())/nthreads;
    info[i].blobmax = ((i+1)*(blobs.size()))/nthreads;
    info[i].mode    = mode;
  }

  for(int i=0;i<nthreads;++i){
    if(pthread_create(threads+i, NULL, t_draw_blobs, info+i)){
      fprintf(stderr, "error creating render thread.\n");
      exit(0);
    }
  }
  for(int i=0;i<nthreads;++i){
    if(int err = pthread_join(threads[i], 0)){
      fprintf(stderr, "error joining render thread. %d\n", err);
      exit(0);
    }
  }


  // tick("draw blobs");
  // printf("done.");

  /////////////////////////////////////////////////////////////////////

  // ** Normalizing shouldn't be necessary, because cellpdf < 1 ** .
  // // get max value.
  // float max = 0;
  // for(int i=0;i<self.w3;i+=2){
  //   if(data[i] > max)max=data[i];
  // }

  // // divide by max value.
  // for(int i=0;i<self.w3;i+=2){
  //   data[i] = (data[i])/(max);
  // }

  ////////////////////////////////////////////////////////////////////

  // scale(0.2f);
  // for(auto blob = blobs.begin(); blob != blobs.end(); ++blob){
  //   ScaleBlob *sb = *blob;
  //   if(sb->parent){
  //     rasterlineadd(vec3(sb->position), vec3(sb->parent->position), 1.1f, 1.5f);
  //   }
  //   // for(ScaleBlob *succ : sb->succ){
  //   //   rasterlineadd(vec3(sb->position), vec3(succ->position), 1.1f, 1.5f);
  //   // }
  // }
  // for(int z=0;z<self.a2; z++){
  //   if(z%50 == 0)printf("draw_blobs progress %d/%d\n",z,self.a2);
  //   for(int y=0; y<self.a1; y++){
  //     for(int x=0; x<self.a0; x++){
  //       int i = x*self.w0 + y*self.w1 + z*self.w2;
  //       vec3 p(x,y,z);
  //       float v= 0.f;
  //       for ( auto blob = blobs.begin(); blob != blobs.end(); ++blob ){
  //         float vv = (*blob)->pdf(p);
  //         if(std::isfinite(vv)){
  //           v += vv;
  //         }
  //       }
  //       data[i] = v;
  //     }
  //   }
  // }
  // normalize(1.f);
  return;

  // if(highlight){
  //   normalize(0.25);
  //   // highlight centers.
  //   float *buff = self.buff[self.curr];
  //   for(int i=0;i<self.w3;i+=self.w0){
  //     buff[i] = buff[i]*0.95f;
  //   }
  //   for(int i=0;i<blobs.size();++i){
  //     glm::ivec3 v(blobs[i]->position);
  //     int xi = v.x*self.w0 + v.y*self.w1 + v.z*self.w2;
  //     if(xi>=0 && xi<self.w3)buff[xi] = 1.f;
  //   }
  // }
}

static inline void fppswap(float **a, float **b){
  float *temp = *a;
  *b          = *a;
  *a          = temp;
}
static inline float sq(float x){
  return x*x;
}

ScaleBlob* model_with_gaussians(){

}

ScaleBlob* ArFilter::compute_blob_tree(){
  BSPTree<ScaleBlob> bspblobs(0,10,vec3(-1.f,-1.f,-1.f),vec3(self.a0+1.f,self.a1+1.f,self.a2+1.f)); // safe against any rounding errors.

  DiscreteKernel kernel; 
  float sigma = 2.f;
  float scale = 0.f;

  float *const iscalec = new float[self.w3];   // data for scaled image.
  float *iscale = iscalec;                     // so we can manipulate this pointer.

  std::vector<ScaleBlob*> blobs;
  float *temp;

  int index_of_laplacian;

  printf("threshold 0.1f;\n");
  threshold(0.1f, 1.f);
  max1();
  // we want to compute the gaussian of the image at various scales.
  // use the fact that the composition of gaussians is a gaussian with
  //  c^2 = a^2 + b^2.
  while(scale < 8.f){

    // printf("filter stack: %p %p. curr=%d.\n", self.buff[0], self.buff[1], self.curr);
    printf("gaussian %.2f -> scale %.2f",sigma, scale);
    kernel = gaussian(sigma, int(sigma*4));   // create gaussian kernel with new sigma.
    set_kernel(kernel);                       //
    filter();                                 // compute gaussian blur, store in self.buff[curr].
    scale = sqrt(scale*scale + sigma*sigma);  // compute scale factor with repeated gaussians.
    // printf("scale = %.2f\n", scale);
    float **blurred = self.buff + self.curr;  // address of blurred image.
    printf("  laplacian.. ");
    laplacian3d(1);                           // compute laplacian, store in self.buff[curr+1].
    // fppswap(blurred, &iscale);                 // remove the blurred image from the swap chain
                                              // and store in iscale, so that it's not overwritten
                                              // by successive filter operations.

    // index_of_laplacian = self.curr;

    temp     = iscale;
    iscale   = *blurred;
    *blurred = temp;

    // printf("swap.\n");
    // printf("filter stack: %p %p. curr=%d.\n", self.buff[0], self.buff[1], self.curr);
    // printf("  normalize.. ");
    filter();
    normalize();
    printf(". find blobs.\n");
    std::vector<ScaleBlob*> bigblobs = find_blobs();
    for(ScaleBlob *sb : bigblobs){
      sb->scale = scale;
      bspblobs.insert(sb, sb->mode);
    }
    // printf("connect..\n");
    for(ScaleBlob *x : blobs){
      // ScaleBlob *closest = 0;
      // float      min_dist = -1;
      // for(ScaleBlob *y : bigblobs){
      //   float dist = sq(x->mode.x - y->mode.x) + sq(x->mode.y - y->mode.y) + sq(x->mode.z - y->mode.z);
      //   if(!closest || dist < min_dist){
      //     min_dist = dist;
      //     closest = y;
      //   }
      // }

      ScaleBlob *closest  = bspblobs.find_closest(x->mode);
      x->parent = closest;
      closest->children.push_back(x);
    }
    bspblobs.clear();

    blobs = bigblobs;

    // prune the tree so that blobs never only have one child.
    // for(int i=0;i<blobs.size();++i){
    //   if(blobs[i]->children.size() == 1){
    //     ScaleBlob *only_child = blobs[i]->children[0];
    //     blobs[i] = only_child;
    //     only_child->parent = 0;
    //     // delete *blobs[i];
    //   }
    // }

    sigma += 1.5f;
    kernel.destroy();

    temp = self.buff[self.curr];
    self.buff[self.curr] = iscale;
    iscale = temp;

    // fppswap(&iscale, self.buff+self.curr);    // put the gaussian blurred image back at self.buff[curr]
                                              // for use in the next filter operation.
    // printf(".done.\n");
  }
  printf("..done.\n");

  // undo our pointer shenanigans.
  // remove the buffer that we just created
  // from the swap chain.
  for(int i=0;i<self.nbuf;++i){
    if(self.buff[i] == iscalec){
      self.buff[i] = iscale;
    }
  }
  // printf("filter stack: %p %p. iscalec=%p.\n", self.buff[0], self.buff[1], iscalec);
  --self.curr;
  if(self.curr<0)self.curr = self.nbuf-1;
  normalize();
  // self.curr = index_of_laplacian;
  // laplacian3d();
  delete[] iscalec;
  ScaleBlob *frame = new ScaleBlob();
  frame->children = blobs;
  frame->parent = 0;
  return frame;
}

BSPTree<ScaleBlob> ArFilter::get_bsp(int depth){
  return BSPTree<ScaleBlob>(0, depth, vec3(-1.f,-1.f,-1.f),vec3(self.a0+1.f,self.a1+1.f,self.a2+1.f));
}

void ArFilter::difference_image(Nrrd* x){
  float *data1 = self.buff[self.curr];
  float *data2 = (float*) x->data;
  for(int i=0;i<self.w3;i+=self.w0){
    data1[i] = fabs(data1[i] - data2[i]);
  }
}