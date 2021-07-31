docker run -it -d --gpus all \
	--name GFE \
	-p 6006:6006 \
	-u 1001:1001 \
	-v /home/yslee/dev/:/WS \
	-e TZ=Asia/Seoul \
	-e python=/opt/conda/bin/python \
	ghcr.io/rapidrabbit76/gnu-srlab-gfe:latest