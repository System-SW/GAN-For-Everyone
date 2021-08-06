# GAN For Everyone repository Rules


## Contact us 
<!-- 관리자 변경시 본인 정보로 수정 바람  -->
| Repository Admin |                                                 |
| ---------------- | ----------------------------------------------- |
| Mail             | yslee.dev@gmail.com                             |
| Github           | [Github Link](https://github.com/rapidrabbit76) |

---


<!-- @TODO: 마지막 업데이트 날짜 수정 확인 -->
## 2021.08.01 
- **Pytorch가 아닌 다른 구현체도 환영합니다.**
- 가능한 [ROOT README](./README.md)의 파일 구조를 지켜주세요.
  - 다른 구조를 사용한다면 모델 README에 설명을 추가해주세요.
```bash
# (option   : 학습 과정 중 생성
# (*)       : 학습에 꼭 필요 혹은 기본 구성요소
RepoRootPath
├── GAN                  # 구현된 모델(구현체)
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
```
- **main** branch는 README를 따르면 실행이 가능한 상태를 유지해 주세요.
  - 로컬 branch는 자유롭게 작성해되 remote branch로 올리지 말아 주세요. 
  - 협업이 필요한 경우 Fork를 통해 별로의 Repo로 작업해주세요.
- [ROOT README](./README.md)의 [Requirements](./requirements.txt)에 포함되어 있지 않은 패키지를 사용할 경우 해당 모델 README에 별도의 Requirements를 만들어 주세요.