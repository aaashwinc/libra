#include <SFML/Graphics.hpp>
#include <teem/nrrd.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "experiment.h"
#include "view.h"
#include "pipeline.h"
#include "synth.h"

class Game{
public:
  View *view;
  ArExperiment *experiment;
  sf::RenderWindow *window;
  sf::View sfview;
  ArPipeline *pipeline;

  sf::Clock clock;
  sf::Font font;
  sf::Text text;

  ReprMode reprmode;
  bool keys[1024];

  float scale  = 0;
  Game() : reprmode("plain"){

  }
  void asserts(bool b, const char *message){
    if(!b){
      fprintf(stderr,"ASSERT: %s\n", message);
      exit(0);
    }
  }
  void initUI(){
    asserts(font.loadFromFile("../rsc/CallingCode-Regular.ttf"), "loading font");
    text.setFont(font);
    text.setString("Loading...");
    text.setCharacterSize(16);
    text.setFillColor(sf::Color::White);
    text.setStyle(sf::Text::Bold);

    window = new sf::RenderWindow(sf::VideoMode(800, 600), "Artemis");
    window->setFramerateLimit(30);

    sfview = window->getDefaultView();
  }
  void init(){
    view->camera.set(vec3(-4,3,6), vec3(1,0,-0.33), vec3(0,0,1));

    // printf("camera: %.2f %.2f %.2f\n",view->camera.right.x,view->camera.right.y,view->camera.right.z);

    experiment = new ArExperiment("/home/ashwin/data/mini/???.nrrd",0,10,4);

    pipeline = new ArPipeline(experiment);
    view->setvolume(pipeline->repr(reprmode));
    for(int i=0;i<1024;i++)keys[i]=false;
  }

  void handle_events(){
      sf::Event event;
      while (window->pollEvent(event)){
        if (event.type == sf::Event::Closed){
          window->close();
        }
        if (event.type == sf::Event::KeyPressed){
          if(event.key.code >= 0){
            keys[event.key.code] = true;
          }
        }
        if (event.type == sf::Event::KeyReleased){
          if(event.key.code >= 0){
            keys[event.key.code] = false;
          }
        }
        if (event.type == sf::Event::Resized){
          window->setView(sfview = sf::View(sf::FloatRect(0,0,window->getSize().x, window->getSize().y)));
        }
      }
  }

  void check_keys(){
    using glm::vec3;
    float speed = 0.1f;

    if(keys[sf::Keyboard::LShift]){
      speed *= 10.f;
    }
    if(keys[sf::Keyboard::W]){
      view->camera.drawflat = false;
      view->move3D(vec3(0,0,speed));
    }
    if(keys[sf::Keyboard::S]){
      view->camera.drawflat = false;
      view->move3D(vec3(0,0,-speed));
    }
    if(keys[sf::Keyboard::A]){
      view->camera.drawflat = false;
      view->move3D(vec3(-speed,0,0));
    }
    if(keys[sf::Keyboard::D]){
      view->camera.drawflat = false;
      view->move3D(vec3(speed,0,0));
    }
    if(keys[sf::Keyboard::R]){
      view->camera.drawflat = false;
      view->move3D(vec3(0,speed,0));
    }
    if(keys[sf::Keyboard::F]){
      view->camera.drawflat = false;
      view->move3D(vec3(0,-speed,0));
    }
    if(keys[sf::Keyboard::Left]){
      view->camera.drawflat = false;
      view->rotateH(0.1f);
    }
    if(keys[sf::Keyboard::Right]){
      view->camera.drawflat = false;
      view->rotateH(-0.1f);
    }
    if(keys[sf::Keyboard::Down]){
      view->camera.drawflat = false;
      view->rotateV(-0.1f);
    }
    if(keys[sf::Keyboard::Up]){
      view->camera.drawflat = false;
      view->rotateV(0.1f);
    }
    if(keys[sf::Keyboard::M]){
      view->camera.flat.slice += 0.04*speed;
      view->camera.drawflat = true;
      view->touch();
    }
    if(keys[sf::Keyboard::N]){
      view->camera.flat.slice -= 0.04*speed;
      view->camera.drawflat = true;
      view->touch();
    }
    if(keys[sf::Keyboard::O]){
      ++reprmode.timestep;
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::P]){
      --reprmode.timestep;
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::Num1]){
      reprmode.name = "plain";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::Num2]){
      reprmode.name = "blobs";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::Num3]){
      reprmode.name = "filter residue";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::Num4]){
      reprmode.name = "filter internal";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::Num5]){
      reprmode.name = "gaussian";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::Num6]){
      reprmode.name = "laplacian";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::Num0]){
      reprmode.name = "sandbox";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::I]){
      reprmode = pipeline->repr_coarser(reprmode);
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::K]){
      reprmode = pipeline->repr_finer(reprmode);
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::G]){
      if(!strcmp(reprmode.geom, "graph")){
        reprmode.geom = "none";
      }else if(!strcmp(reprmode.geom, "none")){
        reprmode.geom = "graph";
      }
      view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::U]){
      pipeline->process(reprmode.timestep,reprmode.timestep+1);
      reprmode.name = "blobs";
      // printf("hello!\n");
      view->setvolume(pipeline->repr(reprmode));
      // printf("bello!\n");
      view->touch();
      // printf("mello!\n");
    }
  }
  void renderall(){
    using std::to_string;
    sf::Time elapsed1 = clock.getElapsedTime();
    static int ms = 0;
    static int renderframenum = 0;
    if(view->render()){
      sf::Time elapsed2 = clock.getElapsedTime();
      double time = (elapsed2.asSeconds() - elapsed1.asSeconds());
      ms = time*1000;
      ++renderframenum;
    }
    text.setString(
        "(rendered " + to_string(renderframenum) +" frames, " + to_string(ms) + "ms)\n" + 
        "timestep "+to_string(reprmode.timestep)+"\n"+
        "render mode: " + std::string(reprmode.name) + "\n" + 
        "scale: " + to_string(reprmode.blob.scale) + "\n" + 
        ((pipeline->get(reprmode.timestep).complete)?"processed":"")
      );

    window->clear(sf::Color(10,10,10));
    view->render_to(window);


    window->draw(text);
    window->display();
  }
  int run(){
    view = new View(350,350);
    init();
    initUI();

    while (window->isOpen()){
      handle_events();
      check_keys();

      renderall();
    }  
  }
};
int main(){  
  Game game;
  return game.run();
}