# =========================================================
# LLM-RAG: Domain-Specific Question Answering System — Makefile
# ---------------------------------------------------------
# On Windows, run these from Git Bash. If `make` is not
# installed, run the underlying commands shown in the README.
# =========================================================

# Resolve the venv Python for the host OS.
ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

.DEFAULT_GOAL := help
.PHONY: help install dev run test docker-build docker-up docker-down clean

help: ## Show available targets
	@echo "LLM Domain-Specific QA — make targets:"
	@echo "  install       Create .venv and install backend + dev dependencies"
	@echo "  dev           Run the API locally with auto-reload (http://127.0.0.1:8000)"
	@echo "  run           Run the API locally without reload (127.0.0.1:8000)"
	@echo "  test          Run the backend test suite"
	@echo "  docker-build  Build the Docker images"
	@echo "  docker-up     Build and start the full stack (frontend on :8080)"
	@echo "  docker-down   Stop and remove the Docker stack"
	@echo "  clean         Remove Python caches"

install: ## Create venv and install dependencies
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt

dev: ## Run the API with auto-reload
	cd backend && ../$(VENV_PY) -m uvicorn app.main:app --reload --port 8000

run: ## Run the API without reload
	cd backend && ../$(VENV_PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

test: ## Run the test suite
	cd backend && ../$(VENV_PY) -m pytest

docker-build: ## Build Docker images
	docker compose build

docker-up: ## Build and start the full stack
	docker compose up --build

docker-down: ## Stop and remove the stack
	docker compose down

clean: ## Remove Python caches
	$(VENV_PY) -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
