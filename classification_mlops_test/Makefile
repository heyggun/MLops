# system python interpreter. used only to create virtual environment
PY = python
VENV = venv
BIN=$(VENV)/bin

# make it work on windows too
ifeq ($(OS), Windows_NT)
    BIN=$(VENV)/Scripts
    PY=python
endif

all: black lint test

$(VENV): requirements.txt
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade -r requirements.txt
	touch $(VENV)

.PHONY: test
test: $(VENV)
	$(BIN)/pytest -rxXs --no-header -vv

.PHONY: lint
lint: $(VENV)
	$(BIN)/flake8

.PHONY: black
black: $(VENV)
	$(BIN)/black .

clean:
	rm -rf $(VENV)
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete