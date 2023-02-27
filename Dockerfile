# syntax=docker/dockerfile:1

FROM dkimg/opencv:4.7.0-ubuntu

RUN apt update -y \
    && apt upgrade -y \
    && apt install ffmpeg -y

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 uninstall opencv_python -y
RUN pip3 install opencv_python --user

COPY ./src .
RUN mkdir ./uploads

CMD [ "python3", "-m" , "flask", "--app", "api_app", "run", "--host=0.0.0.0"]
