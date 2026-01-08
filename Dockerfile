### ---- OG dockerfile ---- 

# FROM python:3.12

# COPY requirements.txt /tmp

# RUN pip install -r /tmp/requirements.txt

# WORKDIR /app

# RUN cd /app && git clone https://github.com/karatedava/ACP-designer-WEB .

# CMD ["python3", "app.py"]

### ---- new dockerfile with LFS instalation ---- 

FROM python:3.12

# Install Git LFS (using the official packagecloud script for Debian/Ubuntu)
RUN apt-get update && \
    apt-get install -y curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

WORKDIR /app

# Clone the repo (Git LFS will now automatically download the lfs files)
RUN cd /app && git clone https://github.com/karatedava/ACP-designer-WEB .

CMD ["python3", "app.py"]