FROM python:3.10-slim

WORKDIR /usr/src/app
COPY . .

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1


RUN pip install --no-cache-dir gradio
RUN pip install -r requirements.txt
RUN playwright install

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python","app.py"]
