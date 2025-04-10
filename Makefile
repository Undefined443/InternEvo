.PHONY: all runtime flash-attention apex megablocks

git-submodule:
	git submodule update --init --recursive

runtime: git-submodule
	pip install -r requirements.txt
	pip install -r requirements/torch.txt
	pip install -r requirements/runtime.txt

flash-attention: runtime
	cd third_party/flash-attention && python setup.py install
	cd third_party/flash-attention/csrc/xentropy && pip install -v .
	cd third_party/flash-attention/csrc/rotary && pip install -v .

apex: runtime
	cd third_party/apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .

megablocks: runtime
	pip install git+https://github.com/databricks/megablocks@v0.3.2

all: runtime flash-attention apex megablocks
