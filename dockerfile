FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current context into the container
COPY . .

# Make the /app directory importable as a Python module path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Default command to run the training script
CMD ["python", "train/train_ppo_standalone.py"]
