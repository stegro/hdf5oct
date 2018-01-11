CXX=h5c++
src= h5read.cc
headers=h5read.h
octs=$(src:.cc=.oct)
objs=$(src:.cc=.o)

H5FLAGS=$(shell octave --eval 'exit(__octave_config_info__ ("build_features").HDF5 != 1)' &> /dev/null && echo "-DHAVE_HDF5") \
$(shell octave --eval 'exit(__octave_config_info__ ("build_features").HDF5_18 != 1)' &> /dev/null && echo "-DHAVE_HDF5_18")

MKOCTFILE=CXX="$(CXX)" CXXFLAGS="-ansi -D_GLIBCXX_USE_CXX11_ABI=0" mkoctfile -v $(H5FLAGS)

VERSION=0.5.0
PACKAGEFILE=hdf5oct-$(VERSION).tar.gz

.PHONY: test clean install uninstall package

all: $(octs) package

%.oct: $(objs)
	echo $(MKOCTFILE) -o $@ $(objs)
	$(MKOCTFILE) -o $@ $(objs)

%.o: %.cc $(headers)
	$(MKOCTFILE) -c $<

clean:
	rm -f *.o *.oct package/inst/* test/test*.h5 $(PACKAGEFILE)

install: $(PACKAGEFILE)
	@echo "-- Install Octave Package ------------"
	octave --texi-macros-file=/dev/null --silent --no-gui --eval "pkg install $(PACKAGEFILE)"

uninstall:
	@echo "-- Uninstall Octave Package ----------"
	octave --silent --no-gui --eval "pkg uninstall hdf5oct"

package: $(PACKAGEFILE)

cp-octave:
	cp h5read.{cc,h} $(HOME)/build/octave/libinterp/dldfcn/

$(PACKAGEFILE): $(octs)
	@echo "-- Create Octave Package Archive ------------"
	mkdir -p package/inst
	cp *.oct package/inst
	tar -czf $(PACKAGEFILE) package/

# TESTING ###########

# a target to test the octave functions
test:
	@echo "-- Perform Tests --------------"
	rm -f test/test*.h5
	cd test && octave --silent --no-gui h5test.m
