sources = RNAGraph

NAME:=hahl
VERSION:=$(shell grep "current_version = " .bumpversion.toml | cut -d' ' -f3 | tr -d '"')
GITHUB:=RNAGraph


.PHONY: test format lint unittest coverage pre-commit clean release
test: format lint unittest

format:
	ruff format $(sources) tests

lint:
	ruff check $(sources) tests
	mypy --config setup.cfg $(sources) tests

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
	rm -rf nul
	rm -rf build
	rm -rf dist

release:
	@echo Releasing: $(VERSION)
	gh release create $(GITHUB) $(VERSION)
