FROM tensorflow/tensorflow:1.14.0
RUN pip install -r requirements.txt
ADD . /var/tf
ENTRYPOINT ["python", "/var/tf/promo_trainer.py"]
