# Use the official Python 3.8.10 runtime as a parent image
FROM python:3.8.10-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Add GPG keys and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    unzip

# Add the Debian buster GPG key
RUN curl -fsSL https://ftp-master.debian.org/keys/archive-key-10.asc | apt-key add - && \
    curl -fsSL https://ftp-master.debian.org/keys/archive-key-10-security.asc | apt-key add - && \
    apt-get update

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the code zip file into the container at /usr/src/app
COPY code.zip .

# copy the data archive
COPY RML2016.10a_dict.pkl .
COPY requirements.txt .

# Extract the zip file and clean up
RUN unzip code.zip && rm code.zip

# Install backend dependencies from Python
RUN pip install -r requirements.txt

# Start the application
CMD ["bash", "-c", "cd RNN && python3 RNN_lstm.py"]
