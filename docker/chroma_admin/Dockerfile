# Web UI for the Chroma vector store
# (https://github.com/flanker/chromadb-admin). The pre-built image on
# Docker Hub (fengzhichao/chromadb-admin) is arm64-only, so build the app
# from the pinned release tag with the upstream Dockerfile's base image.
FROM node:20

WORKDIR /app
# Pinned to a specific release tag; bump deliberately when updating.
RUN git clone --depth 1 --branch 0.0.2 https://github.com/flanker/chromadb-admin.git .

RUN npm install
RUN npm run build

CMD ["npm", "start"]
