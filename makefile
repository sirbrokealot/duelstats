# Makefile
format-black:
	@poetry run black .

format-isort:
	@poetry run isort .

lint-black:
	@poetry run black .

lint-isort:
	@poetry run isort .

lint-pyflakes:
	@poetry run pyflakes .

lint-mypy:
	@poetry run mypy .

lint-mypy-report:
	@mypy ./src --html-report ./mypy_html

format: format-black format-isort

lint: lint-black lint-isort lint-pyflakes lint-mypy

all: format-black format-isort lint-pyflakes lint-mypy