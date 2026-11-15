# Web UI for the Chroma vector store
# (https://github.com/flanker/chromadb-admin). The pre-built image on
# Docker Hub (fengzhichao/chromadb-admin) is arm64-only, so build the app
# from the pinned release tag with the upstream Dockerfile's base image.
FROM node:20

WORKDIR /app

FROM node:20

# Pinned to a specific commit: the latest release tag (0.0.2) uses the
# deprecated Chroma v1 API, while this commit uses the chromadb 2.x JS client
# (v2 API). Bump deliberately when updating.
WORKDIR /app
RUN git clone --depth 1 https://github.com/flanker/chromadb-admin.git . && \
    git checkout efe867c86c78683d90b0eb74b88b351fc08f0b5f  # 2025-11-16

RUN npm install
RUN npm run build

CMD ["npm", "start"]
