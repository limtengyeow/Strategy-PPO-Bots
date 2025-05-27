FROM python:3.11.12

# Create non-root user and set the default user
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
ENV PATH=$PATH:/home/$USERNAME/.local/bin

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir pip --upgrade
RUN pip install --no-cache-dir -r requirements.txt

# Install pre-commit and ipykernel for Jupyter support
RUN pip install --no-cache-dir pre-commit ipykernel

# Optional: Auto-launch Jupyter (comment out if using VS Code)
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
