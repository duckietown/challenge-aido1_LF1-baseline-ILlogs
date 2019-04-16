# Definition of learning from logs container
# We start from tensorflow-gpu image
FROM tensorflow/tensorflow:latest-gpu

# from https://github.com/tensorflow/tensorflow/issues/10776
# this may not be needed if you use pytorch image
RUN cd /usr/local/cuda/lib64 \
    && mv stubs/libcuda.so ./ \
    && ln -s libcuda.so libcuda.so.1 \
    && ldconfig

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
RUN mkdir /workspace
# we make the workspace our working directory
WORKDIR /workspace
RUN mkdir /workspace/data

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
# let's copy all our solution files to our workspace
COPY . .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train CNN model for lane following && Freeze model graph
CMD python src/cnn_training_tensorflow.py && python src/freeze_graph.py
