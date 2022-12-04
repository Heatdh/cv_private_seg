FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip

WORKDIR /app
COPY ./*.py ./
COPY ./src ./src
COPY ./export_model/model_adam_0001_cpu.pth .
COPY requirements.txt ./

RUN pip3 install -r requirements.txt
EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["api.py"]