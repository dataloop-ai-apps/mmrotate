FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.devel.py3.8.pytorch2

USER root

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install openmim

USER 1000

# Install MMCV MMDetection

RUN mim install "mmdet<3.0.0" "mmcv-full"

COPY /requirements.txt .

RUN pip install -r requirements.txt
# Install MMRotate
WORKDIR /mmrotate
RUN git clone https://github.com/open-mmlab/mmrotate.git /mmrotate
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .