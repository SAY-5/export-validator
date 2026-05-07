.PHONY: install build-cpp pipeline test test-cpp test-py lint typecheck clean fmt

VENV ?= .venv
PY    = $(VENV)/bin/python
PIP   = $(VENV)/bin/pip
TORCH_DIR = $(shell $(PY) -c "import torch, os; print(os.path.dirname(torch.__file__))")

install:  ## create venv and install package + dev tools
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

build-cpp:  ## configure & build the C++ comparator and tests
	cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release \
		-DTORCH_DIR=$(TORCH_DIR) \
		$(if $(ONNXRUNTIME_DIR),-DONNXRUNTIME_DIR=$(ONNXRUNTIME_DIR),)
	cmake --build build/cpp -j

pipeline:  ## end-to-end: export resnet18, run both runtimes, write reports
	$(PY) -m export_validator.cli export \
		--model resnet18 \
		--out examples/resnet18.onnx \
		--layer-map examples/resnet18_layer_map.json
	EXPORT_VALIDATOR_BINARY=$(PWD)/build/cpp/export_validator_compare \
	  $(PY) -m export_validator.cli compare \
		--model resnet18 \
		--onnx examples/resnet18.onnx \
		--layer-map examples/resnet18_layer_map.json \
		--report-base examples/reports/resnet18_fp32 \
		--tolerance 1e-4 \
		--backend auto

test: test-cpp test-py  ## run all tests

test-py:  ## run python unit + integration tests
	EXPORT_VALIDATOR_BINARY=$(PWD)/build/cpp/export_validator_compare \
	  RUN_INTEGRATION=1 $(VENV)/bin/pytest

test-cpp:  ## run C++ unit tests
	cd build/cpp && ctest --output-on-failure

lint:  ## ruff + black --check
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/black --check src tests

fmt:  ## ruff --fix + black
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/black src tests

typecheck:  ## mypy strict over src/
	$(VENV)/bin/mypy src

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
	rm -f examples/*.onnx examples/reports/*_pt.evl1 examples/reports/*_ort.evl1 examples/reports/*.layers.txt
