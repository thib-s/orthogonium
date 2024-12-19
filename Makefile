.PHONY: help prepare-dev test test-disable-gpu doc serve-doc
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting on py36, py37, py38"
	@echo "make check_all"
	@echo "       check all files using pre-commit tool"
	@echo "make updatetools"
	@echo "       updatetools pre-commit tool"
	@echo "make test-disable-gpu"
	@echo "       run test with gpu disabled"
	@echo "make serve-doc"
	@echo "       run documentation server for development"
	@echo "make doc"
	@echo "       build mkdocs documentation"

prepare-dev:
	python3 -m pip install virtualenv
	python3 -m venv env_orthogonium
	. env_orthogonium/bin/activate && pip install --upgrade pip
	. env_orthogonium/bin/activate && pip install -e .[dev]
	. env_orthogonium/bin/activate && pre-commit install
	. env_orthogonium/bin/activate && pre-commit install-hooks
	. env_orthogonium/bin/activate && pre-commit install --hook-type commit-msg

test:
	. env_orthogonium/bin/activate && tox

check_all:
	. env_orthogonium/bin/activate && pre-commit run --all-files

updatetools:
	. env_orthogonium/bin/activate && pre-commit autoupdate

test-disable-gpu:
	. env_orthogonium/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox

doc:
	. env_orthogonium/bin/activate && mkdocs build
	. env_orthogonium/bin/activate && mkdocs gh-deploy

serve-doc:
	. env_orthogonium/bin/activate && CUDA_VISIBLE_DEVICES=-1 mkdocs serve
