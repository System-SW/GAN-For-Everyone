# **Table of GANs**

1. **GAN**(Generative adversarial nets), 2014, Paper
2. **[TODO]** **DCGAN**(Deep Convolutional GANs), 2016, Paper
3. **[TODO]** **WGAN**(Wasserstein GAN), 2017, Paper
4. **[TODO]** **WGAN-GP**(WGAN gradient penalty), 2017, Paper
5. **[TODO]** **LSGAN**(Least Square GAN), 2016, Paper
6. **[TODO]** **Pix2Pix**, 2018, Paper
7. TO BE CONTINUED ... 

# **Requirements**

**[TODO]**

# **Tutorial**

- **[TODO]** Repository Rules
- **[TODO]** Anaconda CUDA Env install (Local)
- **[TODO]** Anaconda CUDA Env install (Docker)
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
# (option): 나중에 생성 혹은 불필요 
# (*)			: 학습에 꼭 필요 혹은 기본 구성요소
RepoRootpath
├── 1. GAN					# 구현된 모델(구현체)
│   ├── dataset		 		# downloaded data dir(option)
│   ├── log					# log dir(option) 
│   ├── [model paper].pdf 	# 구현체 원본 논문(option)
│   ├── README.md	 		# 구현체 개별 설명(option)
│   ├── dataset.py 			# 학습 데이터 전처리(*)
│   ├── hyperparameters.py 	# 학습 파라미터(*)
│   ├── model.py			# 구현된 모델(*)
│   ├── train.py			# 구현체 학습자(*)
│   └── opt.py				# Template class 와 같은 부가 요소
├── 2.DCGAN					# 구현된 모델(구현체)
├── N. MyAwesomModel		# 구현된 모델(구현체)
├── README.md
├── ... ETCs
├── ... ETCs
```

## 2. Template Meta

---

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

		- 학습진행필요한 변수를 생성 
		- seed 값 초기화 
		- Tensorboard를 위한 log 디렉토리 생성
		"""

    @abstractmethod
    def train(self):
		""" 
		- 실제 학습을 진행할 메서드 
		- 대부분의 구현은 여기서 이루어 짐 
		"""

    @abstractmethod
    def test(self):
		"""
		- 학습 도중 test/sample 생성을 진행할 메서드
		"""
```

따라서 모든 모델 구현체의 경우 아래와 같은 방식으로 생성되어 있습니다.

```python
# 새로운 모델 구현 예시 (train.py)
class MyAwesomModelTrainer(Template):
	...

if __name__ == '__name__':
	trainer = MyAwesomModelTrainer()
	trainer.train()
```


## 3. Run Model 

---

1. requirements.txt 를 사용해 python 인터프리터의 환경 설정을 진행 합니다.

```bash
you@server:~$ pip install requirements.txt
```

2. 원하는 모델의 디렉토리로 이동해 학습자를 실행합니다. 

```bash
you@server:~$ pwd
# repo root dirpath
you@server:~$ cd MyAwesomModel # 원하는 모델 디렉토리로 이동
you@server:~$ python train.py  # 학습자 실행
```