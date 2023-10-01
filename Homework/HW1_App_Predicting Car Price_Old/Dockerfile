# syntax=docker/dockerfile:1
FROM python

# install app dependencies
# RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install flask==2.1.*
RUN pip install numpy scikit-learn
WORKDIR /app

COPY . /app
# install app


# final configuration

EXPOSE 8000

CMD ["python3", "app.py"]