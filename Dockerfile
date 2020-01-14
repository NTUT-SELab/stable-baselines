ARG PARENT_IMAGE
ARG USE_GPU
FROM $PARENT_IMAGE
ARG INSTALL_MPI

RUN apt-get -y update \
    && apt-get -y install \
    curl \
    cmake \
    default-jre \
    git \
    jq \
    python-dev \
    python-pip \
    python3-dev \
    libfontconfig1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenmpi-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code
ENV VENV /root/venv

COPY ./setup.py /root/code/setup.py
COPY ./stable_baselines/ /root/code/stable_baselines/
RUN \
    pip install virtualenv && \
    virtualenv $VENV --python=python3 && \
    . $VENV/bin/activate && \
    cd $CODE_DIR && \
    pip install --upgrade pip && \
    if [ "$USE_GPU" = "True" ]; then \
        TENSORFLOW_PACKAGE="tensorflow-gpu==1.8.0"; \
    else \
        TENSORFLOW_PACKAGE="tensorflow==1.8.0"; \
    fi; \
    pip install ${TENSORFLOW_PACKAGE} && \
    if [ "$INSTALL_MPI" = "True" ]; then \
        EXTRAS_REQUIRE="[mpi,tests]"; \
    else \
        EXTRAS_REQUIRE="[tests]"; \
    fi; \
    pip install -e .${EXTRAS_REQUIRE} && \
    rm -rf $HOME/.cache/pip && \
    rm -rf stable_baselines

ENV PATH=$VENV/bin:$PATH

# Codacy code coverage report: used for partial code coverage reporting
RUN cd $CODE_DIR && \
    curl -Ls -o codacy-coverage-reporter.jar "$(curl -Ls https://api.github.com/repos/codacy/codacy-coverage-reporter/releases/latest | jq -r '.assets | map({name, browser_download_url} | select(.name | (startswith("codacy-coverage-reporter") and contains("assembly") and endswith(".jar")))) | .[0].browser_download_url')"

CMD /bin/bash
