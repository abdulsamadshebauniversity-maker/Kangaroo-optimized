#---------------------------------------------------------------------
# Kangaroo build (CPU and CUDA)
# Original author: Jean-Luc PONS
# Maintained fork: reproducible targets, configurable CUDA architecture,
# dependency tracking, and compile-time GPU tuning knobs.
#
# Examples:
#   make -j2                          # CPU-only
#   make gpu=1 ccap=86 CUDA=/usr/local/cuda -j2
#   make gpu=1 ccap=86 GPU_GRP_SIZE=64 NB_RUN=64 -j2
#   make test
#
# The CUDA architecture (ccap) must match your GPU. `nvidia-smi` and
# `./kangaroo -l` report the device; CUDA's programming guide maps devices
# to compute capabilities.

CXX       ?= g++
CUDA      ?= /usr/local/cuda
CUDA_LIBDIR ?= $(CUDA)/lib64
CXXCUDA   ?= $(CXX)
NVCC      ?= $(CUDA)/bin/nvcc
ccap      ?= 86
BUILD_DIR ?= obj
TARGET    ?= kangaroo

# Compile-time algorithm knobs. Keep defaults identical to upstream unless
# benchmarking on the target GPU. Both host and device must see the same values.
GPU_GRP_SIZE ?= 128
NB_RUN       ?= 64
GPU_BARRIERS ?= 0

COMMON_CPPFLAGS := -I. -DGPU_GRP_SIZE=$(GPU_GRP_SIZE) -DNB_RUN=$(NB_RUN)
CXXFLAGS        ?= -O2 -g
CXXFLAGS        += -m64 -mssse3 -Wno-unused-result -Wno-write-strings
LDFLAGS         ?=
LDLIBS          := -lpthread

CORE_SRC := SECPK1/IntGroup.cpp main.cpp SECPK1/Random.cpp \
            Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
            SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
            Kangaroo.cpp HashTable.cpp Thread.cpp Check.cpp \
            Backup.cpp Network.cpp Merge.cpp PartMerge.cpp

ifdef gpu
COMMON_CPPFLAGS += -DWITHGPU -I$(CUDA)/include
LDLIBS          += -L$(CUDA_LIBDIR) -lcudart
GPU_OBJ         := $(BUILD_DIR)/GPU/GPUEngine.o
NVCCFLAGS       ?= -O2 -maxrregcount=0 --ptxas-options=-v
NVCCFLAGS       += -m64 -DGPU_GRP_SIZE=$(GPU_GRP_SIZE) -DNB_RUN=$(NB_RUN) -DKANGAROO_USE_BARRIERS=$(GPU_BARRIERS) \
                   -gencode=arch=compute_$(ccap),code=sm_$(ccap) \
                   -gencode=arch=compute_$(ccap),code=compute_$(ccap)
ifdef debug
CXXFLAGS        += -DWITHGPU -g
NVCCFLAGS       += -G -g
endif
endif

OBJ := $(addprefix $(BUILD_DIR)/,$(CORE_SRC:.cpp=.o)) $(GPU_OBJ)
DEP := $(OBJ:.o=.d)

.PHONY: all bsgs clean test bench
all: $(TARGET)
bsgs: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) $(OBJ) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu GPU/GPUEngine.h GPU/GPUMath.h GPU/GPUCompute.h GPU/GPUBatch.h Constants.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -I. -I$(CUDA)/include -c $< -o $@

test: $(TARGET)
	python3 tests/run_tests.py --binary ./$(TARGET)

bench: $(TARGET)
	python3 scripts/benchmark.py --binary ./$(TARGET) --threads 2 --dp 8 --duration 6 --repeats 2

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEP)
