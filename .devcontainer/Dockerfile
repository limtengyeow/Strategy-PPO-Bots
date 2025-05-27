FROM python:3.11.12

# Create non-root user and set the default user
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies as root, system-wide
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir ipykernel

# Install ipykernel system-wide (no --user!)
RUN python -m ipykernel install --name=python3 --sys-prefix

# Switch to non-root user
USER $USERNAME
ENV PATH=$PATH:/home/$USERNAME/.local/bin
