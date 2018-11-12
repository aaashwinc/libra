#include <SFML/Graphics.hpp>
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "experiment.h"
#include "view.h"



void handle_keys(){
  // if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left)){
  //   std::cout << "left" << std::endl;
  // }
}
int main(){
  sf::RenderWindow window(sf::VideoMode(350,350), "SFML works!");

  View view(350,350);
  view.camera.set(vec3(-4,10,7), vec3(1,0,0), vec3(0,0,1));
  Experiment experiment("/home/ashwin/repo/neurons/data/?.nrrd",3,0,0);
  view.setExperiment(&experiment);
  // view.render();
  printf("begin vis\n");
  view.setvolume(0);
  view.raytrace();

  if(true && false){
    sf::Event event;
    double total_time = 0;
    int i=0;
    for(i=0;i<15;i++){
      while (window.pollEvent(event));
      sf::Clock clock;
      sf::Time elapsed1 = clock.getElapsedTime();
      view.raytrace();
      sf::Time elapsed2 = clock.getElapsedTime();
      double time = (elapsed2.asSeconds() - elapsed1.asSeconds());
      total_time += time;
      printf("trial %d %.2f\n", i, time);

      window.clear();
      window.draw(view.getSprite());
      window.display();
    }
    printf("average: %.2f\n", (total_time/i));
    exit(0);
  }
  // printf("hi!\n");

  while (window.isOpen()){
    sf::Event event;
    while (window.pollEvent(event)){
      if (event.type == sf::Event::Closed)
        window.close();
    }

    if(window.hasFocus()){
      bool render = false;
      float speed = 0.1f;
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)){
        speed *= 10.f;
      } 
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::A)){
        render=true;
        view.move(-speed*view.camera.right);
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::D)){
        render=true;
        view.move(speed*view.camera.right);
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::R)){
        render=true;
        view.move(speed*view.camera.up);
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::F)){
        render=true;
        view.move(-speed*view.camera.up);
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::W)){
        render=true;
        view.move(speed*view.camera.look);
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::S)){
        render=true;
        view.move(-speed*view.camera.look);
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::Left)){
        render=true;
        view.camera.set(view.camera.pos, view.camera.look - view.camera.right*0.1f, vec3(0,0,1));
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::Right)){
        render=true;
        view.camera.set(view.camera.pos, view.camera.look + view.camera.right*0.1f, vec3(0,0,1));
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::Down)){
        float dot = glm::dot(view.camera.look,vec3(0,0,1));
        if(dot>-0.9){
          render=true;
          view.camera.set(view.camera.pos, view.camera.look - view.camera.up*0.1f, vec3(0,0,1));
        }
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::Up)){
        float dot = glm::dot(view.camera.look,vec3(0,0,1));
        if(dot<0.9){
          render=true;
          view.camera.set(view.camera.pos, view.camera.look + view.camera.up*0.1f, vec3(0,0,1));
        }
      } 
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::O)){
        render=true;
        view.timestep++;
      }
      if(sf::Keyboard::isKeyPressed(sf::Keyboard::P)){
        render=true;
        view.timestep--;
      }
      if(render){
        view.raytrace();
      }
    }

    window.clear();
    window.draw(view.getSprite());
    window.display();
  }

  return 0;
}