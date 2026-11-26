# LinguaMeet AI - one-command helpers.   `make` or `make help` lists them.
# Real stack, first time:   make setup   then (two terminals)   make gateway   /   make web   then   make tokens
PY      ?= python
PIP     ?= $(PY) -m pip
ROOM    ?= demo
LANGS   ?= en,es,fr,de
ASR     ?= base
PORT    ?= 8080

.DEFAULT_GOAL := help
.PHONY: help install install-torch env models up down check setup gateway web stack tokens demo test test-py test-js bench screenshots clean

help:  ## show this list
	@awk 'BEGIN{FS=":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  make %-14s %s\n",$$1,$$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: ROOM=$(ROOM) LANGS=$(LANGS) ASR=$(ASR) PORT=$(PORT) PY=$(PY)"

install:  ## install the Python package with all extras (nmt, tts, dev)
	$(PIP) install -e ".[nmt,tts,dev]"

install-torch:  ## CPU PyTorch, needed once for the NLLB -> CTranslate2 conversion
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu

env:  ## create .env from .env.example (keys match deploy/livekit.yaml); never overwrites an existing .env
	@if [ -f .env ]; then echo ".env exists - left untouched"; else cp .env.example .env && echo "created .env"; fi

models:  ## download + convert Whisper, NLLB and Piper voices into ./models (needs Hugging Face access)
	$(PY) scripts/download_models.py --langs $(LANGS) --asr-size $(ASR)

up:  ## start the LiveKit server in Docker (Linux, host networking)
	docker compose up -d livekit

down:  ## stop all containers
	docker compose down

check:  ## preflight: settings, models, voices, tokens, LiveKit port (exit 1 on any FAIL)
	$(PY) scripts/check_setup.py --room $(ROOM)

setup: install install-torch env models up check  ## everything above in order, ends with the preflight

gateway:  ## run the translation gateway in ROOM (keep this terminal open)
	$(PY) -m pipeline.server run --room $(ROOM)

web:  ## serve the web client at http://localhost:PORT (keep this terminal open)
	$(PY) -m http.server $(PORT) -d client

stack:  ## everything in Docker: livekit + gateway + web
	docker compose up -d --build

tokens:  ## print join tokens for alice and bruno (paste into two browser windows)
	@echo "alice:"; $(PY) -m pipeline.server token --room $(ROOM) --identity alice
	@echo "bruno:"; $(PY) -m pipeline.server token --room $(ROOM) --identity bruno

demo:  ## standalone demo room with mock models, no LiveKit needed: http://127.0.0.1:8765
	$(PY) -m meetdemo.server

test-py:  ## Python tests
	$(PY) -m pytest -q

test-js:  ## JavaScript client tests (runs npm install first if needed)
	cd client && ( [ -d node_modules ] || npm install ) && npx jest

test: test-py test-js  ## all tests

bench:  ## latency benchmarks -> BENCHMARKS.md (add REAL=asr,nmt,tts when models are installed)
	$(PY) scripts/benchmark.py --runs 5 $(if $(REAL),--real $(REAL),)

screenshots:  ## regenerate demo screenshots and GIFs (needs Playwright + Chromium)
	$(PY) scripts/demo/record_session.py && $(PY) scripts/demo/capture.py && $(PY) scripts/demo/capture_live.py

clean:  ## remove caches
	find . -name __pycache__ -not -path "./client/node_modules/*" -prune -exec rm -rf {} + ; rm -rf .pytest_cache
