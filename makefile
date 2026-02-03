CXX := g++
CXXFLAGS := -std=c++17 -O2 -Igrpc -Isrc
GRPC_FLAGS := $(shell pkg-config --cflags --libs grpc++ protobuf)

SRC := src/main.cpp src/server.cpp
PROTO_SRC := grpc/slm.pb.cc grpc/slm.grpc.pb.cc
BIN := slm

all: $(BIN)

$(BIN):
	$(CXX) $(CXXFLAGS) $(SRC) $(PROTO_SRC) $(GRPC_FLAGS) -lsentencepiece -o $(BIN)

debug:
	$(CXX) -std=c++17 -O0 -g -fsanitize=address,undefined -Igrpc -Isrc \
	$(SRC) $(PROTO_SRC) $(GRPC_FLAGS) -lsentencepiece -o slm_debug

clean:
	rm -f slm slm_debug
