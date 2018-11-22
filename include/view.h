#include <glm/glm.hpp>

#include "experiment.h"

#ifndef VIEW_H
#define VIEW_H

using namespace glm;

class Colormap{
private:
  int nsamples;
  double step;
  float *domain;
  vec4  *range;
  vec4 computecolor(float x);
public:
  Colormap();
  inline vec4 colorof(double x);
};

class Camera{
public:
  Camera();
  float yaw,pitch;
  float lhorz, lvert;
  vec3 pos;
  vec3 look, up, right;
  mat4 screenToWorld;
  void setYawPitch(float y, float p);
  void set(vec3 pos, vec3 look, vec3 up);
};

class View{
public:
  View(int w, int h);
private:
  sf::Uint8 *texdata;
  int mode;
  int const w, h;
  sf::Sprite  sprite;
  sf::Texture texture;
  Experiment *experiment;
  int timestep;
  int rf;

  struct{
    Nrrd *n;
    NrrdAxisInfo *a;
    int w0,w1,w2,w3;
    int    a1,a2,a3;
    short *data;
  }vcache;

  void get_color(float x, sf::Uint8 *color);

public:
  Camera camera;
  Colormap colormap;
  float position;

  float sample(int t, int c, float x, float y, float z, float defaultv=0.f, bool normalize=true);
  float qsample(int c, float x, float y, float z);
  void setvolume(Nrrd *nrrd);
  void render();
  void moveforward(float v);
  void movetime(int n);
  int  gettime();
  void move(vec3 v);
  void turn(float r);
  void raytrace();
  void setExperiment(Experiment *e);
  int get_time();
  long check();
  sf::Sprite &getSprite();
};

#endif