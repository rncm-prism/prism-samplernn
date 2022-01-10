# Pull the base image
FROM tensorflow/tensorflow:latest-gpu

# Create working directory
WORKDIR .

# Copy code and resources
COPY . .

# Install libsndfile (see https://stackoverflow.com/a/62326957/795131)
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

# Install dependencies
RUN pip install numpy==1.20
RUN pip install librosa
RUN pip install natsort

# Run the training script
ENTRYPOINT [ "python", "train.py" ]