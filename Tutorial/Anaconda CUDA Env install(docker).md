![](https://subicura.com/assets/article_images/2017-01-19-docker-guide-for-beginners-1/docker-logo.png)
# Anaconda CUDA Env Install Using Docker

- nvidia-docker를 설치하셔야 합니다. 
- docker 사용법이 아닙니다. 

# Docker Image 직접 빌드
- 편의상 ubuntu base를 사용했습니다. 
- 원하신다면 더 가벼운 이미지를 베이스로 사용하셔도 괜찮습니다.
- Conda 빌드는 원하는 버전으로 변경하시면 됩니다.
  
**DockerFile**
---
```Dockerfile
FROM nvidia/cuda:11.1.1-base-ubuntu20.04
MAINTAINER yslee.dev@gmail.com

# https://repo.anaconda.com/miniconda/
# 링크를 참고해서 원하는 빌드를 사용하세요. 
ARG MINICONDA_DL_URL=https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
ENV CONDA_PATH /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && \
	# INSTALLATION OF PKG
	apt-get install -y \ 
	vim ranger git wget tmux && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	# INSTALLATION OF CONDA
	wget $MINICONDA_DL_URL -O ~/miniconda.sh && \
	/bin/bash ~/miniconda.sh -b -p $CONDA_PATH && \
	ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc && \
	# Python Env Setting
	$CONDA_PATH/bin/conda init bash && \ 
	$CONDA_PATH/bin/pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
	tqdm tensorboard albumentations scipy

WORKDIR /WS

CMD ["/bin/bash"]
```

**Docker Build CLI**
---
```bash
docker build -t test \
  --build-arg MINICONDA_DL_URL=<miniconda dl url> \
  ./Tutorial/docker/
```

**Docker Build Parameters**
---
| Parameter        | Function                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| MINICONDA_DL_URL | miniconda version                                                                                                                    |
|                  | select nimiconda verion in [here](https://repo.anaconda.com/miniconda/)                                                              |
|                  | default version [Miniconda3-py37_4.10.3-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh) |


# Docker Usage

GitHub Container Registry(ghcr)을 통해 빌드된 이미지 내려 받기 
- REPO_ROOT_DIR_PATH는 현제 작업하고 있는 디렉토리로 볼륨을 잡아 주세요. 
- 6006 port는 Tensorboard를 위한 port 입니다. 
- [Container Link](https://github.com/users/rapidrabbit76/packages/container/package/gnu-srlab-gfe)
  
**Docker Run CLI**
---
```bash
docker pull ghcr.io/rapidrabbit76/gnu-srlab-gfe:latest

docker run -it -d --gpus all \
	--name GFE \
	-p <6006>:<6006> \
	-u <1000>:<1000> \
	-v <REPO_ROOT_DIR_PATH>:/WS \
	-e TZ=Asia/Seoul \
	-e python=/opt/conda/bin/python \
	ghcr.io/rapidrabbit76/gnu-srlab-gfe:latest
```

**Docker Run Parameters**
---
| Parameter               | Function                       |
| ----------------------- | ------------------------------ |
| -p 6006                 | tensorboard http web interface |
| -u <1000>:<1000>        | user uid:gid                   |
| -v <REPO_ROOT_DIR_PATH> | you are working directory      |
| -e TZ=Asia/Seoul        | you are Time Zone              |

**UID, GID Check**
---
```bash
you@server:~$ id $USERS
uid=1001(yslee) gid=1001(yslee) groups=1001(yslee),27(sudo),130(docker)
```

