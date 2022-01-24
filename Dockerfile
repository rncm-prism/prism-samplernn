# Pull the base image
FROM tensorflow/tensorflow:latest-gpu

# Create working directory
WORKDIR .

# Copy code and resources
COPY . .

VOLUME /logdir /generated

# Install libsndfile (see https://stackoverflow.com/a/62326957/795131)
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

# This is required so Numba can write to cache under a non-root user
ENV NUMBA_CACHE_DIR=/tmp

# Install dependencies
RUN pip install numpy==1.20
RUN pip install librosa
RUN pip install natsort

# Add a new user "john" with user id 8877
RUN useradd -u 8877 user1
# Change to non-root privilege
USER user1

