#FROM nvidia/cuda:12.6.2-base-ubuntu22.04
FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime
# FROM python:3.12-slim

WORKDIR /model
ENV PYTHONPATH="/model/src"

COPY pyproject.toml .
# Install pip and dependencies
RUN pip install .

# FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-runtime


COPY src src/
COPY data data/
