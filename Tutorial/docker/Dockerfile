

FROM nvidia/cuda:11.1.1-base-ubuntu20.04
MAINTAINER yslee.dev@gmail.com

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
