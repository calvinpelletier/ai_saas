FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN apt-get update

RUN useradd -m fpu
RUN groupadd fpg
RUN usermod -a -G fpg fpu
RUN echo 'alias python=python3' >> /home/fpu/.bashrc

ADD ./env/full/requirements.txt /home/fpu/requirements.txt
RUN pip install -r /home/fpu/requirements.txt

USER fpu
WORKDIR /home/fpu

ENV BASE_CODE_PATH=/home/fpu/code
ENV BASE_DATA_PATH=/home/fpu/data
ENV PYTHONPATH=${BASE_CODE_PATH}:${PYTHONPATH}
ENV GOOGLE_APPLICATION_CREDENTIALS=${BASE_CODE_PATH}/fp/nocommit/google_app_creds.json
