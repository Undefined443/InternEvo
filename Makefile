NVCC_VERSION := $(shell nvcc --version | grep 11.8)
ifeq ($(NVCC_VERSION),)
	$(error "CUDA 11.8 is required")
endif

.PHONY: all runtime flash-attention apex megablocks

all: runtime flash-attention apex megablocks

git-submodule:
	git submodule update --init --recursive

runtime: git-submodule
	pip install -r requirements/torch.txt
	pip install -r requirements/runtime.txt
	pip install -r requirements/ours.txt

flash-attention: runtime
	cd third_party/flash-attention && python setup.py install
	pip install third_party/flash-attention/csrc/xentropy
	pip install third_party/flash-attention/csrc/rotary

apex: flash-attention
	pip install --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" third_party/apex

megablocks: apex
	pip install git+https://github.com/databricks/megablocks@v0.3.2
