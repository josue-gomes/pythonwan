FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/wan \
    && rm -rf /opt/wan/.git

COPY handler.py /handler.py

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/wan
ENV WAN_ROOT=/opt/wan
ENV HF_HOME=/models/hf

WORKDIR /opt/wan
ENTRYPOINT ["python", "-u", "/handler.py"]
