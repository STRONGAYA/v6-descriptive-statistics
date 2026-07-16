# Basic python3 image as base
FROM python:3.10-alpine

# This is a placeholder that should be overloaded by invoking docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="v6-descriptive-statistics"

# Install system dependencies (Alpine uses apk)
RUN apk add --no-cache git

# Install federated algorithm
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir /app

# Set environment variable to make name of the package available within the docker image.
ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `wrap_algorithm()` when the image is run.
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]