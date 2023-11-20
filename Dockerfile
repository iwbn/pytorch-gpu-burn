FROM nvcr.io/nvidia/pytorch:23.08-py3


ARG TEST_DURATION=3600
COPY . /workspace/pytorch-gpu-burn

WORKDIR /workspace/pytorch-gpu-burn

RUN python main.py $TEST_DURATION
