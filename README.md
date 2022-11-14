# 국방 AI 경진대회 코드 사용법
- NavyNet: 김윤수, 성수영, 김덕현, 김민섭


# Abstract
- 이번 대회의 문제개요는 변화 탐지 분야 중 하나로 단순히 변화된 건물 구역을 탐지하는 것 뿐만아니라 변화의 종류를 판단해야하는 분류 문제이다. 우리는 두 이미지를 각각 encoder에 넣어 분류 없이 변화된 구역만 특정하는 binary segmentation 모델을 통해 변화된 건물 구역의 형태를 명확히 잡는 mask_1을 얻고, Baseline을 통해 얻은 분류된 mask_2를 mask_1에서 탐지된 변화 객체들의 변화 종류를 분류하는데 사용하였다.

이는 Baseline과 같이 다른 시간에 찍힌 사진을 양쪽으로 이어붙여 Encoder-Decoder 구조의 모델로 학습할 경우, 변화의 종류는 잘 학습하지만 정확한 위치를 찾는데 있어 다소 미흡한 결과를 보여줬기 때문이다.
이에 착안하여 우리는 새로운 모델의 도입이 필요했고, 분류를 하지않고 정확한 위치를 잡는데 집중할 수 있는 binary segmentation 모델을 사용하게 되었다. 베이스라인은 넓은 receptive field를 통해 이미지의 전체 의미 파악에 용이한 DeepLabV3+를 사용하였다. Binary Segmentation Model로는 UFPN을 사용했는데, 기존 학습 데이터에 y 데이터의 모든 변화 부분을 union하여 하나의 binary 마스크로 만들어 Ground Truth로 사용했다.

# 마스크 합성 로직
- [Fig. 1]과 같이 binary-mask의 각 부분을 베이스라인의 출력을 이용해 채워주는 방식으로 합성이 진행된다.
먼저 opencv의 contour를 사용해 binary-mask에서 연속적인 픽셀들을 하나의 건물로 인식하고 convex hull을 얻는다.
이후 binary-mask에서 얻은 각 건물의 위치에 대해 대응하는 픽셀을 베이스라인의 출력에서 확인하여 최대 빈도로 나타난 class로 전체 영역을 채워준다.

# 핵심 파일 설명
  - 학습 데이터 경로
    - baseline 모델:``
    - change_detection.pytorch 모델:`./my_dataset`
  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터: `./LaMa_models/big-lama-with-discr/models/best.ckpt`
  - 공개 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 3개
    - `./mymodel/models/+.ckpt`
    - `./mymodel/models/last_v10.ckpt`
    - `./mymodel/models/last_v11.ckpt`
  - 학습 실행 스크립트: `./train.sh`
  - 학습 메인 코드: `./bin/train.py`
  - 테스트 실행 스크립트: `./inference.sh`
  - 테스트 메인 코드: `./bin/predict.py`
  - 테스트 이미지, 마스크 경로: `./Inpainting_Test_Raw_Mask`
  - 테스트 결과 이미지 경로: `./final_result/output_aipg`

## 코드 구조 설명
- 베이스라인
    - 최종 사용 모델 : DeepLabV3+
    - Model_encoder : timm-regnetx_120
    - encoder_weight : imagenet
    - loss_weight: [1, 1, 1, 3]
    - input_size : (736, 384)
    -
- binary-mask 생성 코드
    - 최종 사용 모델 : UFPN
    - Model_encoder : ?
    - encoder_weight : ?
- **학습된 가중치 파일(베이스라인 코드) : ?(파일경로/파일이름)**
- **학습된 가중치 파일(binary-mask 생성 코드) : ?(파일경로/파일이름)**
- **최종 제출 파일 : mask.zip**

## 주요 설치 library
- torchmetrics==0.10.0
- torch==1.12.0
- torchvision == 0.1.0
- mmcv-full==1.6.0

# 실행 환경 설정

  - 소스 코드 및 conda 환경 설치
    ```
    unzip military_code.zip -d military_code
    cd ./military_code/detector

    conda env create -f conda_env.yml
    conda activate myenv
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
    pip install pytorch-lightning==1.2.9
    '''
# 학습 실행 방법

  - 학습 데이터 경로 설정
    - `./configs/training/location/my_dataset.yaml` 내의 경로명을 실제 학습 환경에 맞게 수정
      ```
      data_root_dir: /home/user/detection/my_dataset/  # 학습 데이터 절대경로명
      out_root_dir: /home/user/detection/experiments/  # 학습 결과물 절대경로명
      tb_dir: /home/user/detection/tb_logs/  # 학습 로그 절대경로명
      ```

  - 학습 스크립트 실행
    ```
    ./train.sh
    ```
    
  - 학습 스크립트 내용
    ```
    #!/bin/bash

    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    # 첫번째 Model 설정 기반 학습: last_v7__0daee5c4615df5fc17fb1a2f6733dfc1.ckpt, last_v10__dfcb68d46a9604de3147f9ead394f179.ckpt 획득
    python bin/train.py -cn big-lama-aigp-1 location=my_dataset data.batch_size=5

    # 두번째 Model 설정 기반 학습: last_v11__cdb2dc80b605a5e59d234f2721ff80ea.ckpt 획득
    python bin/train.py -cn big-lama-aigp-2 location=my_dataset data.batch_size=5
    ```

# 테스트 실행 방법

  - 테스트 스크립트 실행
    ```
    ./inference.sh
    ```

  - 테스트 스크립트 내용
    ```
    #!/bin/bash

    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    install -d ./final_result/output_aipg

    python bin/predict.py 


    # 상기의 3가지 추론 결과를 Pixel-wise Averaging 처리하여 최종 detection 결과 생성
    python ensemble_avg.py
    ```
