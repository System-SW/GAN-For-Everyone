docker run -it -d --gpus all \
	--name GFE \
	-p 6006:6006 \
	-u 1000:1000 \
	-v <<REPO_ROOT_DIR_PATH>:/WS \
	-e TZ=Asia/Seoul \
	-e python=/opt/conda/bin/python \
	ghcr.io/rapidrabbit76/gnu-srlab-gfe:latest