# Use an official Python runtime as a parent image
FROM python:3.11.10-slim

# Set environment variables to reduce warnings and bugs
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . /app/

# Define the command to run your Scrapy spider
CMD ["scrapy", "crawl", "my_spider"]
