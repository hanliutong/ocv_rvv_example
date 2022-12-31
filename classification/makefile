OCV_ISNTALL_DIR=/root/opencv/build/install

test: build
	/opt/riscv/bin/qemu-riscv64 -cpu rv64,x-v=true,vlen=128 classification.out

build: classification.cpp
	/opt/rvv-llvm/bin/clang++ --static ./classification.cpp -O2 -o classification.out -march=rv64gcv --gcc-toolchain=/opt/riscv --target=riscv64-unknown-linux-gnu --sysroot=/opt/riscv/sysroot -I${OCV_ISNTALL_DIR}/include/opencv4 -L${OCV_ISNTALL_DIR}/lib -lopencv_gapi -lopencv_highgui -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_dnn -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -llibpng -L${OCV_ISNTALL_DIR}/lib/opencv4/3rdparty -llibprotobuf -lade -llibjpeg-turbo -llibwebp -llibtiff -llibopenjp2 -lquirc -llibpng -L/opt/riscv/sysroot/usr/lib -lzlib -ldl -lm -lpthread -lrt

clean:
	rm -rf *.out output.jpg core