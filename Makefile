setup:
	python3.12 -m venv env
	@echo "Environment created"

	env/bin/pip install -r requirements.txt
	@echo "Requirements installed"

install:
	env/bin/pip install -r requirements.txt
	@echo "Requirements installed"

up:
	docker-compose up -d

.PHONY: setup
