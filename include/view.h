#ifndef VIEW_H
#define VIEW_H

#include <glm/glm.hpp>
#include <SFML/Graphics.hpp>
#include "colormap.h"
#include "experiment.h"
#include "shapes.h"

using namespace glm;

class Camera{
public:
  Camera();
  float yaw,pitch;
  float lhorz, lvert;
  vec3 pos;
  vec3 look, up, right;
  vec3 sky;
  mat4 screenToWorld;

  bool drawflat;

  struct{
    float slice;
  }flat;

  void setYawPitch(float y, float p);
  void set(vec3 pos, vec3 look, vec3 up);
};

/* Describes a View into a particular ArPipeline.
 */
class View{
public:
  View(int w, int h);
private:
  ArGeometry3D *geom;

  sf::VertexArray lines;
  // sf::VertexArray lines_window;

  sf::Uint8 *texdata;
  int const w, h;
  sf::Sprite  sprite;
  sf::Texture texture;
  int beat;
  int unstable;

  struct{
    Nrrd *n;
    NrrdAxisInfo *a;
    int w0,w1,w2,w3;
    int    a1,a2,a3;
    float *data;
  }vcache;

  struct{
    int width;
    int height;
  }win;

  void drawflat();
  void draw_geometry();
  void raytrace();
  float qsample(int c, float x, float y, float z);

public:
  Camera camera;
  Colormap colormap;
  float position;

  // render
  void touch();
  int  render();

  // input
  void setvolume(Nrrd *nrrd);
  void setgeometry(ArGeometry3D*);

  // movement
  void move3D(vec3 v);
  void rotateH(float r);
  void rotateV(float r);

  // output
  void render_to(sf::RenderWindow *window);
};

#endif