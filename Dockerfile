FROM python:3.10-slim

WORKDIR /usr/src/app
COPY . .

RUN pip install --no-cache-dir gradio
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["sh", "-c", "playwright install && python app.py"]
