# **Table of GANs**

| Model                            | Year | Paper                                         | State         |
| :------------------------------- | ---- | --------------------------------------------- | ------------- |
| [GAN](./GAN/src/README.pdf)      | 2014 | [PDF](./GAN/src/paper_GAN.pdf)                | Done          |
| [cGAN](./cGAN/README.md)         | 2014 | [PDF](./cGAN/src/paper-Conditional%20GAN.pdf) | **Code Done** |
| [DCGAN](./DCGAN/README.md)       | 2016 | [PDF](./DCGAN/src/paper-DCGAN.pdf)            | Done          |
| [WGAN](./WGAN/README.md)         | 2017 | [PDF](./WGAN/src/paper-WGAN.pdf)              | Done          |
| [WGAN-GP](./WGAN-GP/README.md)   | 2017 | [PDF](./WGAN-GP/src/paper-WGAN-GP.pdf)        | **Code Done** |
| [LSGAN](./LSGAN/README.md)       | 2017 | [PDF](./LSGAN/src/paper-LSGAN.pdf)            | **Code Done** |
| [EBGAN](./EBGAN/README.md)       | 2016 | [PDF](./EBGAN/src/paper-EBGAN.pdf)            | **Code Done** |
| [Pix2Pix](./Pix2Pix/README.md)   | 2016 | [PDF](./Pix2Pix/src/paper-Pix2Pix.pdf)        | **Code Done** |
| [CycleGAN](./CycleGAN/README.md) | 2017 | [PDF](./CycleGAN/src/paper-CycleGAN.pdf)      | **Code Done** |
| [SRGAN](./SRGAN/README.md)       | 2016 | [PDF](./SRGAN/src/paper-SRGAN.pdf)            | **Code Done** |
| [ACGAN](./ACGAN/README.md)       | 2016 | [PDF](./ACGAN/src/paper-ACGAN.pdf)            | **Code Done** |
| [ProGAN](./ProGAN/README.md)     | 2018 | [PDF](./ProGAN/src/paper-ProGAN.pdf)          | **Code Done** |

# **Environment**

```swift
OS: Ubuntu 20.04 LTS x86_64
Kernel: 5.4.0-80-generic
Shell: zsh 5.8
CPU: Intel i9-10980XE (36) @ 4.800GHz
GPU 00: NVIDIA RTX-3090 NVIDIA Corporation Device 2204
GPU 01: NVIDIA RTX-3090 NVIDIA Corporation Device 2204
Memory: 128512MiB
```

# **Requirements**

- albumentations==1.0.3
- PyYAML==5.4.1
- scipy==1.7.0
- tensorboard==2.5.0
- torch==1.9.0
- torchvision==0.10.0
- tqdm==4.61.2

# **Tutorial**

- [Repository Rules](./Rules.md)
- [Anaconda CUDA Env install (Local)](<./Tutorial/Anaconda%20CUDA%20Env%20install(local).md>)
- [Anaconda CUDA Env install (Docker)](<./Tutorial/Anaconda%20CUDA%20Env%20install(docker).md>)
- [Useful Tools](./Tutorial/Tools.md)
- [AMP(Automatic Mixed Precision) package](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Kaggle API(Dataset Download)](./Tutorial/Kaggle%20cli%20tool.md)

# Repository Tutorial

## 0. Index

1. Code Structure
2. Template Meta
3. Run Model

## 1. Code Structure

각 구현체의 경우 디렉터리로 구분되어 있으며 각 구현체 디렉터리의 경우 아래와 같은 구조를 기본으로 합니다. 별도의 구성요소가 포함되면 각 구현체 README.md에 설명을 포함합니다.

```bash
# (option)  : 학습 과정 중 생성
# (*)       : 학습에 꼭 필요 혹은 기본 구성요소
RepoRootPath
│── opt.py                  # Template class 와 같은 부가 요소 link to MyAwesomeModel's
├── dataset                 # 학습 데이터 전처리(*) link to MyAwesomeModel's
├── DATASET                 # downloaded data dir (*)
├── GAN                     # 구현된 모델(구현체)
│   ├── README.md           # 구현체 개별 설명(option)
│   ├── log                 # log dir(option)
│   ├── hyperparameters.py  # 학습 파라미터(*)
│   ├── model.py or module  # 구현된 모델(*)
│   ├── train.py            # 구현체 학습자(*)
│   ├── dataset             # link from root dir's dataset (*)
│   ├── opt.py              # link from root dir's opt.py (*)
│   ├── src
│   │    └── [paper].pdf    # paper of model
├── DCGAN                   # 구현된 모델(구현체)
├── MyAwesomeModel          # 구현된 모델(구현체)
├── README.md
├── ... Etc
```

## 2. Template Meta

**해당 레포는 템플릿 프로그래밍을 사용합니다.**

모든 구현체 내부의 "opt.py"(link) 의 "Template" class를 상속받아 학습을 진행합니다.

Template class 설명은 아래와 같습니다.

```python
class Template(metaclass=ABCMeta):
    """ Abstract Class for Trainer """

    def __init__(self, hp: dict, model_name: str):
        """
        Args:
            hp (dict): hyperparameters for train
            model_name (str): train model name (for log dir name)

        Desc:
        - 학습진행필요한 변수를 생성
        - seed 값 초기화
        - Tensorboard를 위한 log 디렉터리 생성
        """

    @abstractmethod
    def train(self):
        """
        Desc:
        - 실제 학습을 진행할 메서드
        - 대부분의 구현은 여기서 이루어 짐
        """

    def test(self,real):
        """
        Desc:
        - 학습 도중 test/sample 생성을 진행할 메서드
        """
```

따라서 모든 모델 구현체의 경우 아래와 같은 방식으로 생성되어 있습니다.

```python
# 새로운 모델 구현 예시 (train.py)
import hyperparameters as hp
class MyAwesomeModelTrainer(Template):
    def __init__(self):
        super().__init__(hp,'MyAwesomeModel')
        ...
    def train(self):
        ...
    def test(self):
        ...
    ...
if __name__ == '__main__':
    # train.py에 main 작성
    trainer = MyAwesomeModelTrainer()
    trainer.train()
```

## 3. Run Model

1. requirements.txt 를 사용해 python 인터프리터의 환경 설정을 진행 합니다.

```bash
you@server:~$ pip install requirements.txt
```

2. 원하는 모델의 디렉터리로 이동해 학습자를 실행합니다.

```bash
you@server:~$ pwd
[RepoRootPath]
you@server:~$ cd MyAwesomeModel # 원하는 모델 디렉터리로 이동
you@server:~$ python train.py  # 학습자 실행
```

3. TensorBoard를 통해 학습 로그를 시각화합니다.

```bash
you@server:~$ pwd
[RepoRootPath]/MyAwesomeModel
you@server:~$ tensorboard --bind_all --logdir log/MyAwesomModel/[**TimeStamp**]
```
