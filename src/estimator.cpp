#include "estimator.h"
#include <limits>

ScaleBlob Estimator::fit(Nrrd* source, ScaleBlob* in){
  ScaleBlob blob;
  blob.position = in->position;
  blob.min = in->min;
  blob.max = in->max;
  blob.n = in->n;
  blob.shape = in->shape;
  blob.fshape = in->fshape;
  blob.mode = in->mode;
  blob.invCov = in->invCov;

  int a0 = source->axis[0].size;
  int a1 = source->axis[1].size;
  int a2 = source->axis[2].size;


  float bestalpha;
  float bestbeta;
  float bestkappa;
  float besterror = std::numeric_limits<float>::infinity();

  float error = 0;
  std::vector<ivec3> indices;
  float n = 0;
  for (int xx=blob.min.x; xx < blob.max.x; xx++){
    for(int yy=blob.min.y; yy < blob.max.y; yy++){
      for(int zz=blob.min.z; zz < blob.max.z; zz++){
         vec3 p(xx,yy,zz);
         if(blob.ellipsepdf(p) > 0){
          indices.push_back(ivec3(xx,yy,zz));
          ++n;
        }
      }
    }
  }
  float beta = 1.f;
  // for(float beta = 0.33; beta <=1.33; beta+=0.33f){
    for(float kappa = 0.1; kappa < 0.95; kappa += 0.05f){
      for(float alpha=0.16f; alpha < 0.5f; alpha*= 1.41421f){
        
        // for each beta, kappa, alpha:
        blob.kappa = kappa;
        blob.alpha = alpha;
        blob.beta  = beta;
        error = 0;
        for(ivec3 v : indices){
          vec3 p(v);
          int xx = v.x;
          int yy = v.y;
          int zz = v.z;
          int i = xx + source->axis[0].size*yy + source->axis[0].size*source->axis[1].size*zz;
          float diff = ((float*)source->data)[i] - blob.generalized_multivariate_gaussian_pdf(p);
          error += diff * diff;
        }
        error /= n;
        // printf("  error = %.5f\n", error);
        if(error<besterror){
          besterror = error;
          bestalpha = alpha;
          bestbeta  = beta;
          bestkappa = kappa;
        }
      }
    }
  // }
  blob.alpha = bestalpha;
  blob.beta = bestbeta;
  blob.kappa = bestkappa;
  printf("best parameters: %.2f %.2f %.2f, err=%.5f\n", bestalpha, bestbeta, bestkappa, besterror);

  if (error < 1 ){
    while(blob.generalized_multivariate_gaussian_pdf(blob.min) > 0.00001){
      blob.min -= vec3(10,10,10);
      if(blob.min.x <= 0 && blob.min.y <= 0 && blob.min.z <= 0 )break;
    }
    while(blob.generalized_multivariate_gaussian_pdf(blob.max) > 0.00001){
      blob.max += vec3(10,10,10);
      if(blob.max.x >= a0 && blob.max.y >= a1 && blob.max.z >= a2 )break;
    }
  }
  // blob.min.x = blob.position.x
  return blob;
}