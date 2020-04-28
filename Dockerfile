FROM python:3.8-slim

ARG dataset

RUN mkdir /app
COPY ./ /app/

WORKDIR /app
RUN pip install -r requirements.txt
RUN python train.py ${dataset} --evaluate

CMD ["/bin/bash"]
