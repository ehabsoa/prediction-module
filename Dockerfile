FROM python:3.6

WORKDIR /opt/bunching

COPY . /opt/bunching/

ENTRYPOINT python "startServer.py"
