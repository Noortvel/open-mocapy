# syntax=docker/dockerfile:1

FROM dkimg/opencv:4.7.0-ubuntu

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./src .

CMD [ "python3", "-m" , "flask", "--app", "api_app", "run", "--host=0.0.0.0"]
