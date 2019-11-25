FROM tensorflow/tensorflow:1.14.0-py3
ADD . /var/tf
WORKDIR /var/tf
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT ["python3", "/var/tf/promo_trainer.py"]
