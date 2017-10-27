NVCC = /usr/local/cuda-9.0/bin/nvcc
FLAGS = -O3 -std=c++14 -arch=sm_61 -rdc=true -Xcompiler="-Wall"

all: example

example: example.cu
	$(NVCC) $(FLAGS) example.cu -o example

run: example
	./example

clean: example
	rm -rf example
