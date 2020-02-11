
EIGEN=/usr/local/include/eigen3
LBFGSpp=/home/ashwin/repo/libra/lib/LBFGSpp/include
LBFGS=$(BUILD_DIR)/../lib/lbfgs
LIBXML2=/usr/include/libxml2/
GLAD=/home/ashwin/repo/voxels/lib/glad/include

BUILD_DIR = $(shell pwd)
SRC_DIR = $(BUILD_DIR)/../src

GLFW = $(BUILD_DIR)/../lib/glfw-3.2.1

CC=gcc -O4
CXX=g++ -g -O4
RM=rm -f

IPATH    = -I$(VULKAN_SDK)/include -I$(GLFW)/include -I${CURDIR}/../include -I$(EIGEN) -I$(LBFGSpp) -I$(LBFGS)/include -I$(LIBXML2) -I$(GLAD)
RPATH    = -Wl,-rpath,$(GLFW)/build/src
CPPFLAGS = -g $(IPATH)
LDFLAGS  = -g -L$(GLFW)/build/src 
LDLIBS   =
LIBS     = -lteem -lsfml-graphics -lsfml-window -lsfml-system -lglfw3 -lGL -lX11 -lXi -lXrandr -lXxf86vm -lXinerama -lXcursor -lrt -lm -pthread -ldl -lpng -lz -lxml2
CLIBS    = -lm

SRCS=$(wildcard $(SRC_DIR)/*.cpp)
DEPS=$(wildcard $(SRC_DIR)/*.d)
CSRCS=$(wildcard $(SRC_DIR)/*.c)
OBJS=$(subst .cpp,.o,$(SRCS))


all: game

game: $(OBJS) lbfgs
	$(CXX) $(CPPFLAGS) $(IPATH) $(LDFLAGS) -o game $(LBFGS)/lbfgs.o $(OBJS) $(LDLIBS) $(RPATH) $(LIBS)

lbfgs: $(LBFGS)/lbfgs.o

$(LBFGS)/lbfgs.o: $(LBFGS)/lbfgs.c
	# HELLO!
	$(CC) -c -I$(LBFGS)/include $< -o $(LBFGS)/lbfgs.o -lm 

%.o: %.cpp
	$(RM) $*.d
	$(CXX) $(IPATH) $(CPPFLAGS) -MM $*.cpp -MT $*.o > $*.d
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $*.cpp -o $*.o

# depend: .depend

# .depend: $(SRCS)
# 	# DEPEND
# 	$(RM) ./.depend
# 	$(CXX) $(IPATH) $(CPPFLAGS) -MM $^ -MT $(OBJS) >>./.depend;

clean:
	$(RM) $(OBJS)
	$(RM) $(LBFGS)/lbfgs.o

distclean: clean
	$(RM) *~ .depend

include $(DEPS)