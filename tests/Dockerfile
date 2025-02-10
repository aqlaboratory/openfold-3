FROM openfold-base

COPY tests /opt/openfold3/tests
COPY run_pretrained_openfold.py /opt/openfold3/run_pretrained_openfold.py
COPY train_openfold.py /opt/openfold3/train_openfold.py
COPY environments/development.txt /opt/openfold3/development.txt

WORKDIR /opt/openfold3
RUN pip install -r development.txt
