#include "view.h"

Camera::Camera(){
  lhorz = 1;
  lvert = 1;
  yaw   = 0;
  pitch = 0;

  pos   = vec3();
  look  = vec3();
  up    = vec3();
  right = vec3();
  sky   = vec3();
  
  screenToWorld = mat4();
  drawflat   = false;

  flat.slice = 0;
}
void Camera::set(vec3 pos, vec3 look, vec3 sky){
  this->pos = pos;
  this->look = normalize(look);
  this->right = normalize(cross(look, sky));
  this->up = cross(right, look);
  this->sky = sky;
}