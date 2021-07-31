# **Table of GANs**
| No   | Model                         | Title                                                                                        | Year | Paper                                                                                                                                        |
| :--- | :---------------------------- | :------------------------------------------------------------------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | [GAN](./1.GAN/src/README.pdf) | Generative adversarial nets                                                                  | 2014 | [PDF](./1.GAN/src/NIPS-2014-generative-adversarial-nets-Paper.pdf)                                                                           |
| 2    | [DCGAN](./2.DCGAN/README.md)  | Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks | 2016 | [PDF](./2.DCGAN/src/DCGAN(UNSUPERVISED%20REPRESENTATION%20LEARNING%20WITH%20DEEP%20CONVOLUTIONAL%20GENERATIVE%20ADVERSARIAL%20NETWORKS).pdf) |
| 3    | **[TODO]** **WGAN**           |                                                                                              |      | [PDF]()                                                                                                                                      |
| 4    | **[TODO]** **WGAN-GP**        |                                                                                              |      | [PDF]()                                                                                                                                      |
| 5    | **[TODO]** **LSGAN**          |                                                                                              |      | [PDF]()                                                                                                                                      |
| 6    | **[TODO]** **Pix2Pix**        |                                                                                              |      | [PDF]()                                                                                                                                      |
| 7    | **[TODO]** **LSGAN**          |                                                                                              |      | [PDF]()                                                                                                                                      |
<!-- 3. (Wasserstein GAN), 2017, Paper
4. **[TODO]** **WGAN-GP**(WGAN gradient penalty), 2017, Paper
5. **[TODO]** **LSGAN**(Least Square GAN), 2016, Paper
6. **[TODO]** **Pix2Pix**, 2018, Paper
7. TO BE CONTINUED ... -->

# **Environment**
```swift
OS: Ubuntu 20.04 LTS x86_64 
Kernel: 5.4.0-80-generic 
Shell: zsh 5.8 
CPU: Intel i9-10980XE (36) @ 4.800GHz 
GPU: NVIDIA 68:00.0 NVIDIA Corporation Device 2204 
GPU: NVIDIA 1a:00.0 NVIDIA Corporation Device 2204 
Memory: 128512MiB 
```
# **Requirements**
- albumentations==1.0.3 
- PyYAML==5.4.1
- scipy==1.7.0
- tensorboard==2.5.0
- torch==1.9.0 
- torchaudio==0.9.0a0+33b2469
- torchvision==0.10.0
- tqdm==4.61.2

# **Tutorial**

- **[TODO]** Repository Rules
- [Anaconda CUDA Env install (Local)](./Tutorial/Anaconda%20CUDA%20Env%20install(local).md)
- [Anaconda CUDA Env install (Docker)](./Tutorial/Anaconda%20CUDA%20Env%20install(docker).md)
- **[TODO]** GPU Monitoring Tools
- **[TODO]** AMP(Automatic Mixed Precision) package
- **[TODO]** Kaggle cli tool(Dataset Download)

# Repository Tutorial

## 0. Index

1. Code Structure
2. Template Meta
3. Run Model

## 1. Code Structure

각 구현체의 경우 디렉터리로 구분되어 있으며 각 구현체 디렉터리의 경우 아래와 같은 구조를 기본으로 합니다. 별도의 구성요소가 포함되면 각 구현체 README.md에 설명을 포함합니다.

```bash
# (option   : 학습 과정 중 생성
# (*)       : 학습에 꼭 필요 혹은 기본 구성요소
RepoRootPath
├── 1. GAN                  # 구현된 모델(구현체)
│   ├── dataset             # downloaded data dir(option)
│   ├── log                 # log dir(option)
│   ├── log.tar.gz          # 비교를 위한 사전 학습 로그 (option)
│   ├── [model paper].pdf   # 구현체 원본 논문(option)
│   ├── README.md           # 구현체 개별 설명(option)
│   ├── dataset.py          # 학습 데이터 전처리(*)
│   ├── hyperparameters.py  # 학습 파라미터(*)
│   ├── model.py            # 구현된 모델(*)
│   ├── train.py            # 구현체 학습자(*)
│   └── opt.py              # Template class 와 같은 부가 요소
├── 2.DCGAN                 # 구현된 모델(구현체)
├── N. MyAwesomModel        # 구현된 모델(구현체)
├── README.md
├── ... ETCs
```

## 2. Template Meta

**해당 레포는 템플릿 프로그래밍을 사용합니다.**

모든 구현체 내부의 "opt.py" 의 "Template" class를 상속받아 학습을 진행합니다.

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

    @abstractmethod
    def test(self):
        """
        Desc:
        - 학습 도중 test/sample 생성을 진행할 메서드
        """
```

따라서 모든 모델 구현체의 경우 아래와 같은 방식으로 생성되어 있습니다.

```python
# 새로운 모델 구현 예시 (train.py)
import hyperparameters as hp
class MyAwesomModelTrainer(Template):
    def __init__(self):
        super().__init__(hp,'MyAwesomModel')
        ...
    def train(self):
        ...
    def test(self):
        ...

if __name__ == '__main__':
    # train.py에 main 작성
    trainer = MyAwesomModelTrainer()
    trainer.train()
```


## 3. Run Model

**GPU를 사용한 학습을 위해서는 CUDA를 설치해야 합니다.**

**Tutorial을 확인하시고 CUDA 설치를 진행하세요.**

1. requirements.txt 를 사용해 python 인터프리터의 환경 설정을 진행 합니다.

```bash
you@server:~$ pip install requirements.txt
```

2. 원하는 모델의 디렉터리로 이동해 학습자를 실행합니다.

```bash
you@server:~$ pwd
[RepoRootPath]
you@server:~$ cd MyAwesomModel # 원하는 모델 디렉터리로 이동
you@server:~$ python train.py  # 학습자 실행
```

3. TensorBoard를 통해 학습 로그를 시각화합니다.

```bash
you@server:~$ pwd
[RepoRootPath]/MyAwesomModel
you@server:~$ tensorboard --bind_all --logdir log/MyAwesomModel/[TimeStamp]
```