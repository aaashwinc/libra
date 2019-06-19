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
#include <LBFGS.h>

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
  bool clicked[3];
  ivec2 mousepos;

  float scale  = 0;
  Game() : reprmode("plain"){
    // ScaleBlob b1;
    // ScaleBlob b2;
    // b1.position = vec3(0,0,0);
    // b1.covariance << 1, 0, 0,
    //                  0, 3, 0,
    //                  0, 0, 0.25;
    
    // b2.position = vec3(4,0,0);
    // b2.covariance << 2, 0, 0,
    //                  0, 1, 0,
    //                  0, 0, 1;
    // printf("init\n");
    // b1.distance(&b2);

    // exit(0);

  }
  void asserts(bool b, const char *message){
    if(!b){
      fprintf(stderr,"ASSERT: %s\n", message);
      ::exit(0);
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
  void save(){
    FILE *file = fopen("../rsc/save.artemis","wb");
    fwrite(&(view->camera),sizeof(view->camera),1,file);
    fclose(file);

    pipeline->save();
  }
  void load(){
    FILE *file = fopen("../rsc/save.artemis","rb");
    fread(&(view->camera),sizeof(view->camera),1,file);
    fclose(file);

    pipeline->load();
  }
  void exit(){
    save();
    window->close();
  }
  void init(){
    // synth();
    view->camera.set(vec3(-4,3,6), vec3(1,0,-0.33), vec3(0,0,1));

    // printf("camera: %.2f %.2f %.2f\n",view->camera.right.x,view->camera.right.y,view->camera.right.z);

    experiment = new ArExperiment("/home/ashwin/data/miniclear/???.nrrd",0,20,4);

    pipeline = new ArPipeline(experiment);
    view->setvolume(pipeline->repr(reprmode));
    for(int i=0;i<1024;i++)keys[i]=false;
    for(int i=0;i<3;i++)clicked[i]=false;

    // pipeline->process(reprmode.timestep,reprmode.timestep+3);

    load();

    reprmode.name = "blobs";
    reprmode.geom = "graph";
    view->setvolume(pipeline->repr(reprmode));
    view->setgeometry(pipeline->reprgeometry(reprmode));
    view->touch();

    // todo: remove

    // pipeline->process(reprmode.timestep,reprmode.timestep+1);
    
    // pipeline->load();
    // ::exit(0);
    ///////////////
  }

  void handle_events(){
    sf::Event event;
    while (window->pollEvent(event)){
      if (event.type == sf::Event::Closed){
        exit();
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
      if (event.type == sf::Event::MouseMoved){
        mousepos.x = event.mouseMove.x;
        mousepos.y = event.mouseMove.y;
      }
      if (event.type == sf::Event::MouseButtonPressed){
        if (event.mouseButton.button == sf::Mouse::Left){
          clicked[0] = true;
        }
        if (event.mouseButton.button == sf::Mouse::Right){
          clicked[1] = true;
        }
      }
      if (event.type == sf::Event::MouseButtonReleased){
        if (event.mouseButton.button == sf::Mouse::Left){
          clicked[0] = false;
        }
        if (event.mouseButton.button == sf::Mouse::Right){
          clicked[1] = false;
        }
      }
    }
  }

  void check_keys(){
    using glm::vec3;
    float speed = 0.1f;

    if(keys[sf::Keyboard::F2]){
      exit();
    }
    if(keys[sf::Keyboard::LShift]){
      speed *= 10.f;
    }
    if(keys[sf::Keyboard::LControl]){
      speed *= 0.1f;
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
    if(keys[sf::Keyboard::Dash]){
      view->step_gamma(1.1f);
      view->touch();
    }
    if(keys[sf::Keyboard::Equal]){
      view->step_gamma(1/1.1f);
      view->touch();
    }
    if(keys[sf::Keyboard::LBracket]){
      view->step_falloff(1.1f);
      view->touch();
    }
    if(keys[sf::Keyboard::RBracket]){
      view->step_falloff(1/1.1f);
      view->touch();
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
      if(!keys[sf::Keyboard::LShift]){
        keys[sf::Keyboard::O] = false;
      }
      --reprmode.timestep;
      view->setvolume(pipeline->repr(reprmode));
      view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::P]){
      if(!keys[sf::Keyboard::LShift]){
        keys[sf::Keyboard::P] = false;
      }
      ++reprmode.timestep;
      view->setvolume(pipeline->repr(reprmode));
      view->setgeometry(pipeline->reprgeometry(reprmode));
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
    if(keys[sf::Keyboard::Num7]){
      reprmode.name = "blobs_succs";
      view->setvolume(pipeline->repr(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::I]){
      reprmode = pipeline->repr_coarser(reprmode);
      view->setvolume(pipeline->repr(reprmode));
      view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::B]){
      printf("pos: %.3f %.3f %.3f\n", view->camera.pos.x, view->camera.pos.y, view->camera.pos.z);
      keys[sf::Keyboard::B] = false;
    }
    if(keys[sf::Keyboard::K]){
      reprmode = pipeline->repr_finer(reprmode);
      view->setvolume(pipeline->repr(reprmode));
      view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::G]){
      if(!strcmp(reprmode.geom, "none")){
        reprmode.geom = "paths";
      }else if(!strcmp(reprmode.geom, "paths")){
        reprmode.geom = "graph";
      }else if(!strcmp(reprmode.geom, "graph")){
        reprmode.geom = "succs";
      }else{
        reprmode.geom = "none";
      }
      view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
      keys[sf::Keyboard::G] = false;
    }
    if(clicked[0]){
      printf("clicked %d %d\n", mousepos.x, mousepos.y);
      pipeline->repr_highlight(&reprmode, view->camera.pos*33.f, view->pixel_to_ray(vec2(mousepos)), keys[sf::Keyboard::LControl], keys[sf::Keyboard::LShift]);
      view->setvolume(pipeline->repr(reprmode));
      view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
      clicked[0] = false;
    }
    if(keys[sf::Keyboard::U]){
      pipeline->process(reprmode.timestep,reprmode.timestep+1);
      reprmode.name = "blobs";
      reprmode.geom = "graph";
      view->setvolume(pipeline->repr(reprmode));
      view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
    }
    if(keys[sf::Keyboard::L]){
      pipeline->find_paths();
      // pipeline->process(reprmode.timestep,reprmode.timestep+2);
      // // reprmode.name = "blobs";
      // reprmode.geom = "graph";
      // view->setvolume(pipeline->repr(reprmode));
      // view->setgeometry(pipeline->reprgeometry(reprmode));
      view->touch();
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
        "render mode: " + std::string(reprmode.name) + " - " + std::string(reprmode.geom) + "\n" + 
        "scale:   " + to_string(reprmode.blob.scale) + "\n" + 
        "gamma:   " + to_string(view->get_gamma()) + "\n" + 
        "falloff: " + to_string(view->get_falloff()) + "\n" + 
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
  setvbuf(stdout, NULL, _IONBF, 0);  
  Game game;
  return game.run();
}