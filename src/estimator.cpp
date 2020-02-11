#include "estimator.h"
#include <limits>
#include <random>


static float randf(){
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}
static float randf(float min, float max){
  return min + randf()*(max-min);
}
ScaleBlob Estimator::fit(Nrrd* source, ScaleBlob* in){
  ScaleBlob blob;
  blob.position = in->position;
  blob.min = in->min;
  blob.max = in->max;
  blob.n = in->n;
  blob.shape = in->shape;
  blob.mode = in->mode;
  blob.invCov = in->invCov;

  int a0 = source->axis[0].size;
  int a1 = source->axis[1].size;
  int a2 = source->axis[2].size;


  float bestalpha = 0;
  float bestbeta = 0;
  float bestkappa = 0;
  float besterror = std::numeric_limits<float>::infinity();

  float error = 0;
  std::vector<ivec3> indices;
  int n = 0;
  printf("indices..");
  for (int xx=0; xx < a0; xx++){
    for(int yy=0; yy < a1; yy++){
      for(int zz=0; zz < a2; zz++){
         vec3 p(xx,yy,zz);
         if(blob.ellipsepdf(p) > 0){
          indices.push_back(ivec3(xx,yy,zz));
          n = n+1;
        }
      }
    }
  }
  if(n == a0*a1*a2){
    indices = std::vector<ivec3>();
    indices.push_back(ivec3(blob.position.x, blob.position.y, blob.position.z));
  }
  // printf("max = %d\n",a0*a1*a2);
  printf("estimate.. n=%.2f, ", n);
  float beta = 1.f;
  for(int i=0;i<1000;i++){
    // printf("%d, ", i);
    float beta  = randf(1,6);
    float kappa = randf(0.1,0.95);
    float alpha = randf(0.15, 0.5);

  // for(float beta = 1.f; beta < 6.f; beta *= 1.2f){
  //   // printf("beta = %.2f. ", beta);
  //   for(float kappa = 0.1; kappa < 0.95; kappa += 0.05f){
  //     for(float alpha=0.08f; alpha < 0.5f; alpha*= 1.1){
        // float r1 = 
        // float beta = 1.f;
        // for each beta, kappa, alpha:
        blob.model.kappa = kappa;
        blob.model.alpha = alpha;
        blob.model.beta  = beta;
        error = 0;
        // printf("hi . ");
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
    //   }
    // }
  }
  blob.model.alpha = bestalpha;
  blob.model.beta = bestbeta;
  blob.model.kappa = bestkappa;
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
  in->min = blob.min;
  in->max = blob.max;
  in->model.alpha = blob.model.alpha;
  in->model.beta = blob.model.beta;
  in->model.kappa = blob.model.kappa;
  in->model.type = 'g';

  // blob.min.x = blob.position.x
  return blob;
}