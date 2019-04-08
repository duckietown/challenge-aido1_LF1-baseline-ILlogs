repo=aidonode-aido2_lf-scenario_maker
# repo=$(shell basename -s .git `git config --get remote.origin.url`)
branch=$(shell git rev-parse --abbrev-ref HEAD)
tag=duckietown/$(repo):$(branch)

build:
	docker build -t $(tag) .

build-no-cache:
	docker build -t $(tag)  --no-cache .

push: build
	docker push $(tag)

test1:
	python3 scenario_maker.py < test_data/in.json
