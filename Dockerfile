FROM zauberzeug/l4t-tkdnn-darknet:latest

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV LANG C.UTF-8

RUN python3 -m pip install --upgrade pip
# fixing pyYAML upgrade error (see https://stackoverflow.com/a/53534728/364388)
RUN python3 -m pip install --no-cache-dir --ignore-installed PyYAML
# installing dependencies
RUN python3 -m pip install --no-cache-dir "uvicorn[standard]" async_generator aiofiles psutil
RUN python3 -m pip install --no-cache-dir "learning-loop-node==0.3.9" 
RUN python3 -m pip install --no-cache-dir retry debugpy pytest-asyncio icecream pytest autopep8

RUN git clone https://git.hipert.unimore.it/fgatti/darknet.git
RUN cd darknet && \
    make && \
    mkdir layers debug

WORKDIR /app/

COPY ./start.sh /start.sh
ADD ./converter /app
ENV PYTHONPATH=/app
EXPOSE 80
ENV HOST=learning-loop.ai
ENV LANG C.UTF-8

CMD ["/start.sh"]