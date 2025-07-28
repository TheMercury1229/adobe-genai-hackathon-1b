FROM --platform=linux/amd64 python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy all folders that start with "Collection" directly into /app
COPY Collection*/ ./

# Default command to run the new pipeline
CMD ["python", "app/main.py"]
