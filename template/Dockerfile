# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY ./* ./

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "--server.maxUploadSize=1024", "--server.maxMessageSize 2048","app_streamlit_2.py","--server.port=8080"]