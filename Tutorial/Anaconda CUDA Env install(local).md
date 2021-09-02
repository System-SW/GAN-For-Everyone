# Anaconda CUDA Env Install

1. 아래의 명령어를 입력해 주세요.
   1. 사용하고자 하는 CUDA 및 torch 버전에 따라 명령어가 다릅니다.
   2. https://pytorch.org/ 를 통해 명령어를 확인해 주세요.
   3. 설명에서 진행하는 버전의 경우 torch:1.9.0, cuda:11.1입니다.

```bash
# pip를 사용하는 설치가 아닌 conda 설치이기 때문에 conda env에 cudatoolkit이 적용됩니다.
you@server:~$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
Collecting package metadata (current_repodata.json): done
Solving environment: |
   ...
  typing_extensions  pkgs/main/noarch::typing_extensions-3.10.0.0-pyh06a4308_0
  zstd               pkgs/main/linux-64::zstd-1.4.9-haebb681_0


Proceed ([y]/n)? y
   ...
done

```

2. 설치가 완료되었다면 인터프리터를 실행 시켜 torch가 정상적으로 동작하는지 확인해 주세요.

```bash
you@server:~$ python
Python 3.8.10 (default, Jun  4 2021, 15:09:15)
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__version__
'1.9.0'
>>> torch.zeros([100,100,100,100]).to('cuda:0')
...
         [[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]]], device='cuda:0')
>>>

```

3. device='cuda:0'으로 확인 된다면 설치가 완료되었습니다.
