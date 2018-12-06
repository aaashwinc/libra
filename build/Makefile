
EIGEN=/usr/local/include/eigen3


BUILD_DIR = $(shell pwd)
SRC_DIR = $(BUILD_DIR)/../src

GLFW = $(BUILD_DIR)/../lib/glfw-3.2.1

CC=gcc
CXX=g++ -O4
RM=rm -f

IPATH    = -I$(VULKAN_SDK)/include -I$(GLFW)/include -I${CURDIR}/../include -I$(EIGEN)
RPATH    = -Wl,-rpath,$(GLFW)/build/src
CPPFLAGS = -g $(IPATH)
LDFLAGS  = -g -L$(GLFW)/build/src
LDLIBS   =
LIBS     = -lteem -lsfml-graphics -lsfml-window -lsfml-system -lglfw3 -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -pthread -ldl -lpng -lz

SRCS=$(wildcard $(SRC_DIR)/*.cpp)
OBJS=$(subst .cpp,.o,$(SRCS))


all: game

game: $(OBJS)
	$(CXX) $(IPATH) $(LDFLAGS) -o game $(OBJS) $(LDLIBS) $(RPATH) $(LIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(IPATH) $(CPPFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) *~ .depend

include .depend