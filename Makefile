.PHONY: clean clean-build clean-pyc clean-test dist docs help install lint lint/flake8 lint/black
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -f .coverage-junit.xml
	rm -f coverage.xml

lint/flake8: ## check style with flake8
	flake8 src/analyticon_fifa tests

lint/black: ## check style with black
	black --check src/analyticon_fifa tests

lint: lint/flake8 lint/black ## check style with flake8 and black

test/mypy: ## tests with mypy
	mypy

test/pytest: ## tests with pytest
	pytest

test: test/mypy test/pytest ## runs mypy and pytest

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/analyticon_fifa.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ src/analyticon_fifa
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python -m build
	ls -l dist

install-dev: clean ## installs the package's dev dependencies
	pip install -e .[dev]

install-docs: clean ## installs the docs dependencies
	pip install -e .[docs]

install: clean ## install the package with the -e flag
	pip install -e .
