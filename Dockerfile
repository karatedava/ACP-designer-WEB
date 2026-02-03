### ---- OG dockerfile ---- 

# FROM python:3.12

# COPY requirements.txt /tmp

# RUN pip install -r /tmp/requirements.txt

# WORKDIR /app

# RUN git clone https://github.com/karatedava/ACP-designer-WEB .

# # solve the issue with folder permissions #
# RUN mkdir -p static/runs/GENERATE && \
#     mkdir -p static/runs/MUTATE && \
#     mkdir -p src/models/generative/progen2-ACP-inference && \
#     chmod -R 777 static && \
#     chmod -R 777 src

# CMD ["python", "download_models.py"]

### ---- NEW dockerfile ---- 

FROM python:3.12

# Install deps as root
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

# Clone repo
RUN git clone https://github.com/karatedava/ACP-designer-WEB .

# Create needed directories (as root)
RUN mkdir -p static/runs/GENERATE \
    && mkdir -p static/runs/MUTATE \
    && mkdir -p src/models/generative/progen2-ACP-inference \
    && chmod -R 777 static \
    && chmod -R 777 src

# Optional: make sure huggingface cache is also writable
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

# Run download script → then your real app (via entrypoint)
COPY download_models.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]