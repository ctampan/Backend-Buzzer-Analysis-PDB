FROM godatadriven/pyspark:latest

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT [ "spark-submit" ]

CMD [ "app.py" ]