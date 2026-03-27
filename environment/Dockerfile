# hash:sha256:595c9eeb50d6c2fa7156b3a12f0743fa97fb84f0969dfd7832458a7eaebe8bc2
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# 1. 시스템 패키지 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git-lfs \
        libgl1-mesa-glx \
        libglib2.0-0 \
        lmodern \
        pandoc \
        python-dev \
        tesseract-ocr \
        texlive-xetex \
        libgomp1 \
        libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 3.9 가상환경 생성 및 기본 설정
RUN conda create -n paddle_env python=3.9 -y
ENV PATH=/opt/conda/envs/paddle_env/bin:$PATH

# 3. Python 패키지 설치
# 수정사항: numpy<2.0 (NumPy 2.0 충돌 방지)
RUN pip install -U --no-cache-dir \
    absl-py \
    asttokens \
    astunparse \
    backcall \
    blinker \
    boto3 \
    botocore \
    cachetools \
    certifi \
    chardet \
    charset-normalizer \
    click \
    cmake \
    contourpy \
    cycler \
    decorator \
    executing \
    filelock \
    flatbuffers \
    fonttools \
    gast \
    gensim \
    grpcio \
    h5py \
    idna \
    imageio \
    imageio-ffmpeg \
    importlib-metadata \
    importlib-resources \
    ipython \
    itsdangerous \
    jedi \
    jinja2 \
    jmespath \
    joblib \
    keras \
    keras-applications \
    keras-preprocessing \
    kiwisolver \
    lazy-loader \
    libclang \
    lit \
    llvmlite \
    markdown \
    markupsafe \
    matplotlib \
    matplotlib-inline \
    mdutils \
    more-itertools \
    moviepy \
    mpmath \
    networkx \
    numba \
    "numpy<2.0" \
    oauthlib \
    opencv-python-headless \
    opt-einsum \
    packaging \
    pandas \
    pandoc \
    parso \
    pexpect \
    pickleshare \
    pillow \
    plumbum \
    ply \
    proglog \
    prompt-toolkit \
    protobuf \
    ptyprocess \
    pure-eval \
    pyasn1 \
    pyasn1-modules \
    pygments \
    pypandoc \
    pyparsing \
    pytesseract \
    pytz \
    pywavelets \
    pyyaml \
    regex \
    requests \
    requests-oauthlib \
    requests-unixsocket \
    rsa \
    s3transfer \
    scikit-image \
    scikit-learn \
    scipy \
    six \
    smart-open \
    stack-data \
    sympy \
    tensorboard \
    tensorboard-data-server \
    tensorboard-plugin-wit \
    tensorflow \
    tensorflow-estimator \
    tensorflow-io-gcs-filesystem \
    termcolor \
    threadpoolctl \
    tifffile \
    tiktoken \
    torch \
    tqdm \
    traitlets \
    triton \
    typing-extensions \
    tzdata \
    urllib3 \
    wcwidth \
    werkzeug \
    wrapt \
    zipp \
    paddlepaddle==2.6.2 \
    paddleocr==2.7.3 \
    ruamel.yaml