FROM python:3.12-slim

WORKDIR /model

ENV PYTHONPATH="$PYTHONPATH:/model/src"
COPY pyproject.toml .
RUN pip install .

COPY src src/
COPY data data/
