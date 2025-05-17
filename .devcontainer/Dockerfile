FROM python:3.10-slim

WORKDIR /workspace/Strategy-PPO-Bots

RUN apt-get update && apt-get install -y git curl build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch stable-baselines3 pandas

CMD ["bash"]
