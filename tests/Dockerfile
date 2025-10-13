FROM openfold-base

COPY tests /opt/openfold3/tests
COPY environments/development.txt /opt/openfold3/development.txt
COPY scripts/download_openfold_params.sh /opt/openfold3/download_openfold_params.sh

WORKDIR /opt/openfold3
RUN pip install -r development.txt
