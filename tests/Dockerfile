FROM openfold-base

COPY tests /opt/openfold3/tests
COPY environments/development.txt /opt/openfold3/development.txt

WORKDIR /opt/openfold3
RUN pip install -r development.txt
