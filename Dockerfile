FROM python:3.9.16-bullseye

WORKDIR /usr/src/TweetAnalyzer
COPY app/ ./
COPY data/ ./

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /usr/src/TweetAnalyzer/app
CMD ["python", "web_server.py"]