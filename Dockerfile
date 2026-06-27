FROM python:3.12-slim AS runtime

ARG INSTALL_EXTRAS="audio,video,av,web"
ARG INCLUDE_DENSEAV=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AV_TOOLBOX_CACHE_DIR=/cache/av_toolbox

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[${INSTALL_EXTRAS}]" \
    && if [ "$INCLUDE_DENSEAV" = "1" ]; then \
        python -m pip install --no-cache-dir -e ".[denseav]"; \
       fi

VOLUME ["/cache/av_toolbox", "/app/outputs", "/app/data_segments"]
EXPOSE 8501

CMD ["av-toolbox", "serve", "--host", "0.0.0.0", "--port", "8501"]
