windows:
	nvcc main.cpp blur.cu -o ejecutable_windows/blur -O3 -lgdi32 -luser32 -lshell32

linux:
	nvcc -std=c++11 -arch=sm_60 -o ejecutable_linux/blur main.cpp blur.cu -O3 -L/usr/X11R6/lib -lm -lpthread -lX11 -lnvToolsExt
