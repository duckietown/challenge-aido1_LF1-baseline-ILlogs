ARG AIDO_REGISTRY

FROM ${AIDO_REGISTRY}/duckietown/challenge-aido_lf-template-tensorflow:daffy-amd64

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}


# let's create our workspace, we don't want to clutter the container
RUN rm -r /workspace; mkdir /workspace

WORKDIR /workspace

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
# IMPORTANT: This should fail if "frozen_graph.pb" does not exist.
# This file will contain a learned model which the agent should execute.
COPY frozen_graph.pb .
COPY . .


# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# necessary to get opencv to work properly
RUN apt-get update && apt-get install -y libsm6 libxext6

# CMD python solution.py
ENTRYPOINT ["python", "imitation_agent.py"]
