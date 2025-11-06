FROM python:3.12

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

WORKDIR /app

RUN cd /app && git clone https://github.com/karatedava/ACP-designer-WEB .

CMD ["python3", "app.py"]