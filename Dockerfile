FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN pip install Cython==0.28.4
RUN pip install scikit-image==0.15.0
RUN pip install tqdm==4.56.0
RUN python3 -m pip install pycocotools==2.0.0
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

COPY . .
