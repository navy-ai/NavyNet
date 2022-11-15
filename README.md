# 국방 AI 경진대회 코드 사용법
- NavyNet: 김윤수, 성수영, 김덕현, 김민섭


# Abstract
이번 대회의 문제개요는 변화 탐지 분야 중 하나로 단순히 변화된 건물 구역을 탐지하는 것 뿐만아니라 변화의 종류를 판단해야하는 분류 문제이다. 우리는 두 이미지를 각각 encoder에 넣어 분류 없이 변화된 구역만 특정하는 Binary Segmentation 모델을 통해 변화된 건물 구역의 형태를 명확히 잡는 binary mask를 얻고, Baseline을 통해 얻은 분류된 classification mask를 binary mask에서 탐지된 변화 객체들의 변화 종류를 분류하는데 사용하였다.

이는 Baseline과 같이 다른 시간에 찍힌 사진을 양쪽으로 이어붙여 Encoder-Decoder 구조의 모델로 학습할 경우, 변화의 종류는 잘 학습하지만 정확한 위치를 찾는데 있어 다소 미흡한 결과를 보여줬기 때문이다. 이에 착안하여 우리는 새로운 모델의 도입이 필요했고, 분류를 하지않고 정확한 위치를 잡는데 집중할 수 있는 Binary Segmentation 모델을 사용하여 baseline의 classification mask의 건물 윤곽을 보완했다. 베이스라인은 넓은 receptive field를 통해 이미지의 전체 의미 파악에 용이한 DeepLabV3+를 사용하였다. Binary Segmentation 모델로는 UFPN을 사용했는데, 기존 학습 데이터에 y 데이터의 모든 변화 부분을 union하여 하나의 binary 마스크로 만들어 Ground Truth로 사용했다.

# 마스크 합성 로직
먼저 데이터를 Baseline model에 통과시켜 classification mask를, Binary Segmentation모델에 통과시켜 binary mask를 얻는다. opencv connectedComponents함수를 이용해서 binary mask에서 각 건물의 픽셀 위치 집합을 추출한다. 추출한 픽셀 위치에 있는 classification mask 상의 변화 class 값들 중에서 빈도수가 가장 큰 class를 binary mask의 해당 건물의 변화 class로 지정해준다. 만약 binary mask에서 추출된 건물 객체 위치에 classification mask의 변화 class가 0으로 배경으로 판단 될때는 갱신 class로 설정한다. 이 이유는 Baseline 모델은 class 신축(1) class 소멸(2)에 관한 검출은 각각 0.9 이상의 mIoU성능을 보이지만 class 갱신(3)의 판별에 낮은 성능을 보여 이를 보완하기 위함이다.

# 핵심 파일 설명
  - pre_processing.ipynb      전처리 실행 파일
  - post_processing.ipynb     후처리 실행 파일
  
  - BaselineModel/            베이스라인 모델
    - config/
      - predict.yaml          추론 설정 config 파일
      - train.yaml            학습 설정 config 파일
    - results/
      - pred/                 추론 결과 저장 폴더
      - train/                학습 결과 저장 폴더
    - predict.py              추론 실행 파일
    - train.py                학습 실행 파일
    
  - BinarySegmentationModel/  이중분할 모델
    - train.py                학습 실행 파일
    - infer.py                추론 실행 파일
    - res/                    추론 결과 저장 폴더
    - save/                   학습 데이터 저장 폴더
  
  - test/
    - x/                      베이스라인 테스트 이미지 데이터 폴더
    - input1/                 이중분할 모델 테스트 왼쪽 이미지 데이터 폴더
    - input2/                 이중분할 모델 테스트 오른쪽 이미지 데이터 폴더
  - train/
    - x/                      베이스라인 학습 이미지 데이터 폴더
    - y/                      베이스라인 학습 타겟 데이터 폴더
    - input1/                 이중분할 모델 학습 왼쪽 이미지 데이터 폴더
    - input2/                 이중분할 모델 학습 오른쪽 이미지 데이터 폴더
    - binary_mask/            이중분할 모델 학습 타겟 데이터 폴더
    - val/                    이중분할 모델 검증 데이터 폴더
       - input1               이중분할 모델 검증 왼쪽 이미지 데이터 폴더
       - input2               이중분할 모델 검증 오른쪽 이미지 데이터 폴더
       - binary_mask          이중분할 모델 검증 타겟 데이터 폴더
  
  - res/                      테스트 결과 폴더
    - aggregated_res/         두 모델 테스트 합성 결과 마스크 폴더 
    - final_res/              합성 마스크 후처리 결과 폴더 (최종 결과 폴더)
    - final_rgb/              최종 결과 파일 시각화 폴더
       
  - 공개 Pretrained 모델 기반으로 추가 학습을 한 파라미터 파일
    - 베이스라인 모델
       - ./BaselineModel/results/train/SoTa/model.pt
    - 이중분할 모델
       - ./BinarySegmentationModel/save/epoch_sota(31).pth
       
  - 전처리 파일 : pre_processing.ipynb
  - 베이스라인 모델 학습 실행 스크립트: `python ./BaselineModel/train.py`
  - 베이스라인 모델 추론 실행 스크립트: `python ./BaselineModel/predict.py`
  - 이중분할 모델 학습 실행 스크립트: `python ./BinarySegmentationModel/train.py`
  - 이중분할 모델 추론 실행 스크립트: `python ./BinarySegmentationModel/infer.py`
  - 후처리 파일 : post_processing.ipynb

## 코드 구조 설명
- Baseline
  ### model (./BaselineModel/config/train.yaml 19~25)
  ```
  architecture: DeepLabV3Plus
  encoder: timm-regnetx_120 #timm-regnety_016
  encoder_weight: imagenet #imagenet, noisy-student
  depth: 5
  n_classes: 4
  activation: null
  ```
  ### model (./BaselineModel/config/train.yaml 31~34)
  ```
  loss: 
  name: MeanCCELoss
  args:
    weight: [1, 1, 1, 3]
  ```
- BinarySegmentationModel
  ### model (./BinarySegmentationModel/train.py 34~44)
  ```
  model = UFPNnet(
    encoder_name="timm-regnety_320",
    encoder_weights='imagenet',
    in_channels=3,
    classes=2,
    decoder_attention_type="eca",
    siam_encoder=True,
    decoder_channels=[384, 256, 128, 64, 32],
    fusion_form='diff',
    head='cond',
  )
  ```
  ### loss (./BinarySegmentationModel/train.py 67~68)
  ```
  ce_weight = torch.tensor([1.0, 2.0]).to(DEVICE)
  loss = cdp.utils.losses.MultiHeadCELoss(weight=ce_weight, loss2=True, loss2_weight=1.0)
  ```

- **학습된 가중치 파일(Baseline) : ./BaselineModel/results/train/SoTa/model.pt**
- **학습된 가중치 파일(BinarySegmentationModel) : ./BinarySegmentationModel/save/epoch_sota(31).pth**
- **최종 제출 파일 :./sota_final_res.zip**
### Pretrained 
* Baseline
  * segmentation-models-pytorch 모듈을 사용
* BinarySegmentation
  * timm 모듈 사용(0.4.12 버전 확인) 

## 주요 설치 library
- pytorch==1.7.1
- torchvision==0.8.2
- torchaudio==0.7.2
- cudatoolkit=11.0
- segmentation-models-pytorch==0.3.0
- timm==0.4.12 ## 버전 주의
- albumentations==1.3.0
- opencv-python==4.6.0.66
- scikit-learn==1.0.2

# 실행 환경 설정

  - 소스 코드 및 conda 환경 설치
    ```
    conda env create -f requirements.yaml
    conda activate maicon

    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
    ```

# 전처리 실행 방법

  - ./pre_processing.ipynb 실행
  
  - 아래와 같은 디렉토리 구조 생성

        '''
        # BEFORE
        root_dir
            -- test
                --x
            -- train
                --x
                --y

        # AFTER
        root_dir
            -- test
                --input1
                --input2
                --x
            -- train
                --input1
                --input2
                --binary_mask
                --x
                --y
                --val
                    --input1
                    --input2
                    --binary_mask

        '''
        
  - 모든 셀 실행

# 학습 실행 방법
## Baseline
### 학습 데이터 경로 설정
`./BaselineModel/train.py` 내의 경로명을 실제 학습 환경에 맞게 수정
```
train_dirs: '../train'  # 학습 데이터 상대경로 64번째 줄
train_result_dir: './BaselineModel/results/train'  # 학습 결과물 상대경로 53번째 줄
logger: './BaselineModel/results/train'  # 학습 로그 절대경로명 58번째 줄
```
    

학습 실행
```
cd /workspace/Final_Submission/BaselineModel
python train.py
```
## BinarySegmentation
### 학습 데이터 경로 설정
`./BinarySegmentationModel/train.py` 내의 경로명을 실제 학습 환경에 맞게 수정
```
# 학습 데이터 상대경로 46~61
train_dataset = PRCV_CD_Dataset(
  '../train/',              
  sub_dir_1='input1',
  sub_dir_2='input2',
  img_suffix='.png',
  ann_dir='../train/binary_mask',
  size=256,
  debug=False
)

valid_dataset = PRCV_CD_Dataset(
  '../train/val',
  sub_dir_1='input1',
  sub_dir_2='input2',
  img_suffix='.png',
  ann_dir='../train/val/binary_mask',
  size=256,
  debug=False,
  test_mode=True
)
                                
'./BinarySegmentationModel/save'  # 학습 결과물 상대경로 130번째 줄
```
    

학습 실행
```
cd /workspace/Final_Submission/BinarySegmentationModel
python train.py
```

# 테스트 실행 방법
## Baseline
테스트 스크립트 실행
```
cd /workspace/Final_Submission/BaselineModel
python predict.py

# 결과 파일 경로 /workspace/Final_Submission/BaselineModel/results/pred
```

## BinarySegmentation
```
cd /workspace/Final_Submission/BinarySegmentationModel
python infer.py

# 결과 파일 경로 /workspace/Final_Submission/BinarySegmentationModel/res
```

# 후처리 실행 방법

  - ./post_processing.ipynb 실행
  
  - 첫번째 셀 (마스크 합성) 아래 경로 확인 후 실행
      src_BSM             # BinarySegmentationModel 추론결과 경로
      src_baseline        # BaselineModel 추론결과 경로
      save_path           # 마스크 합성 결과 저장 경로
      ```
      1. Baseline mask와 BinarySegmentationModel 합치기
            src_BSM = './BinarySegmentationModel/res/'
            src_baseline= './BaselineModel/results/pred/SoTa_20221115_021934/mask/'
            save_path = './res/aggregated_res/'
      ...
      ```
  - 두번째 셀 (마스크 후처리) 아래 경로 확인 후 실행
      save_path           # 마스크 합성결과 저장 경로 
      save_to_pat         # 마스크 후처리 결과 저장 경로
      ```
      2. 일정 개수 이하의 라벨은 모두 제거하기
            save_path = './res/aggregated_res/'
            save_to_path = './res/final_res/'
      ...
      ```
  - 세번째 셀 (마스크 시각화) 아래 경로 확인 후 실행
      final_res_path      # 마스크 후처리 결과 저장 경로
      rgb_final_res_path  # 마스크 시각화 결과 저장 경로
      ```
      3. rgb로 결과 확인하기
            final_res_path = './res/final_res/'
            rgb_final_res_path = './res/final_rgb/'
      ...
      ```