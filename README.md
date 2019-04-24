 # AI Driving Olympics

<a href="http://aido.duckietown.org"><img width="200" src="https://www.duckietown.org/wp-content/uploads/2018/12/AIDO_no_text-e1544555660271-1024x638.png"/></a>


## Tensorflow baseline "Solution template" for lane following challenge `aido-LF`

This is a baseline solution for one of the challenges in the [the AI Driving Olympics](http://aido.duckietown.org/).

The [online description of this challenge is here][online].

For submitting, please follow [the instructions available in the book][book].

Note: In particular you will need to set a token and specify your docker username before submitting.

[book]: http://docs.duckietown.org/DT19/AIDO/out/

[online]: https://challenges.duckietown.org/v4/humans/challenges/aido2-LF-sim-validation

## Description

In this baseline template, driving behavior is learned using imitation learning from `rosbag` logs.

Follow the makefile in the individual folders of this repository.

1. Download and extract data using the Makefile in the `extract_data` folder.
2. Learn a model using the Makefile and scripts in the `learning` folder.
3. Submit the learned model by running `make submit` in the `imitation_agent` folder.

Each folder has its own README for further instructions.

This code has been tested on Mac OS and Ubuntu 16.04 both running Python 3.7 and Python 2.7.12.
