repo=challenge-aido_LF-baseline-IL-logs-tensorflow
# repo=$(shell basename -s .git `git config --get remote.origin.url`)
branch=$(shell git rev-parse --abbrev-ref HEAD)
tag=duckietown/$(repo):$(branch)

build:
	docker build -t $(tag) .

build-no-cache:
	docker build -t $(tag)  --no-cache .

push: build
	docker push $(tag)

# test-data1-direct:
	# ./random_agent.py < test_data/in1.json > test_data/out1.json

# test-data1-docker:
	# docker run -i $(tag) < test_data/in1.json > test_data/out1.json


submit:
	dts challenges submit

# --------- Installing Duckietown Gym locally for fast testing -------------
# Run "python eval.py" to locally evaluate your model after installing gym-duckietown
gym-duckietown:
		git clone https://github.com/duckietown/gym-duckietown.git; \
		cd gym-duckietown; \
		pip3 install --user --no-cache -e .;