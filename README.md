# 전통문양 감성 분류 - 이미지 + 텍스트 융합 앙상블

전통문양 이미지에 감성 형용사를 다는 태깅 모델임. 22개 어휘에 대한 multi-label classification 이고, 흑백 문양 이미지 한 장과 그 이미지에 대한 설명 문장을 받아 형용사 5개를 출력함. 정답 형용사도 항상 5개이고 전문가가 붙인 것임. 정답과 예측이 모두 5개이므로 precision = recall = F1 이고, 지표는 F1@5 임.

```
F1@5 = |예측 ∩ 정답| / 5
```

이 저장소는 이미지 인코더 하나와 텍스트 인코더 다섯 개를 각각 융합한 모델 5종을 학습하고, 그 확률을 가중합해 앙상블하는 코드임.

설계 근거와 실험 기록은 저장소가 아니라 프로젝트 노트에 있음.
→ [전통문양 감성 라벨 예측](https://chb2066.github.io/projects/traditional-patterns/)

## Models

val 446장 기준. 다섯 모델 모두 동일한 frozen DINOv3 feature 와 융합한 뒤 fine-tune 함.

학습된 가중치는 배포하지 않음. 아래 표는 학습 결과 기록이고, 재현은 `train/` 의 학습 코드로 하면 됨. HuggingFace 백본은 표에 적힌 이름으로 자동으로 받아짐.

| Model | HuggingFace | val F1@5 |
|---|---|---:|
| kobigbird | `monologg/kobigbird-bert-base` | 0.7861 |
| klue | `klue/roberta-large` | 0.7794 |
| xlmr | `xlm-roberta-large` | 0.7753 |
| mbert | `bert-base-multilingual-cased` | 0.7753 |
| kcbert | `beomi/kcbert-large` | 0.7682 |
| **Ensemble (5)** | | **0.8000** |

## 구성

**이미지 - DINOv3 (동결)**

ConvNeXtV2, SigLIP2, FG-CLIP2, DINOv3 를 비교했을 때 이미지 단독 성능이 가장 좋았음. 학습 중 가중치를 갱신하지 않고 이미지 feature 를 뽑는 역할로만 씀. feature 는 미리 뽑아 두므로 학습 중에 이미지 픽셀을 다시 읽지 않음.

**텍스트 - 언어모델 5종 (fine-tune)**

입력은 설명 문장 하나뿐임. 문양유형·재질·시대 같은 구조적 메타데이터는 넣지 않음.

**융합 - 2토큰 self-attention**

이미지 벡터와 텍스트 벡터를 같은 1024차원으로 맞추고, 둘을 토큰 2개짜리 시퀀스로 취급해 self-attention(Q=K=V)을 한 번 적용함. 그 출력으로 22개 라벨 점수를 내고 상위 5개를 고름.

**앙상블**

5개 모델의 확률을 가중합함. 가중치는 val 446장 기준 Dirichlet 랜덤서치로 탐색한 값이며 `inference/config.json` 의 `ensemble_weight` 에 들어 있음.

## 설치

```bash
pip install -r inference/requirements.txt
```

HuggingFace 백본은 최초 실행 시 자동으로 내려받아 캐시함(총 5~6GB, 인터넷 연결 필요).

## 데이터

데이터 위치는 환경변수로 지정함.

```bash
export PATTERN_DATA_ROOT=/path/to/dataset
```

학습 코드가 기대하는 레코드 구성.

| 필드 | 설명 |
|---|---|
| 이미지 | 흑백 선화 문양 이미지 파일 |
| `description` | 문화포털이 원래 제공하는 이미지 설명 데이터. 학습에 쓰는 유일한 텍스트 입력 |
| 감성 라벨 | 22개 어휘 중 5개. 전문가가 붙인 정답 태그 |

`dataset.py` 의 `load_records()` 가 split 을 만들 때 이미지 파일의 존재 여부를 확인하므로, 동일한 3,954 / 446 split 을 재현하려면 원본 파일 자체는 있어야 함.

데이터셋 자체는 ETRI 과제로 구축된 것이라 포함하지 않음. 텍스트 입력으로 쓰는 `description` 은 문화포털이 제공하는 설명 데이터이므로, 문화포털 등에 공개된 전통문양 이미지와 그 설명을 그대로 넣어 추론해 볼 수 있음.

## 학습

```bash
python train/train_fusion_final.py --desc_only   # 설명 문장만 사용해 모델 1개 학습
python train/ensemble.py                         # 5개 가중치 탐색
python val/analyze_val.py                        # val 결과 분석
```

## 추론

학습으로 만든 가중치를 `inference/weights/` 에 두면 데이터셋 없이 동작함. 가중치 파일명은 `inference/config.json` 에 적힌 대로 `klue.pt`, `xlmr.pt`, `kcbert.pt`, `mbert.pt`, `kobigbird.pt` 임.

```bash
cd inference
python infer.py --image <이미지경로> --description "<설명 문장>"
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--image` | 입력 이미지 경로 | 필수 |
| `--description` | 설명 문장. 메타데이터 없이 순수 서술만 | 필수 |
| `--device` | `cuda` / `cpu` | 자동 |

```
Top-5 예측:
  고전적인: 0.9301
  조화로운: 0.9059
  소박한: 0.8168
  단순한: 0.7357
  우아한: 0.7333
```

코드에서 직접 쓰려면

```python
from infer import EnsembleModel
model = EnsembleModel()
labels, scores = model.predict(image_path, description, topk=5)
```

## 데모

이미지와 `description` 을 넣으면 top-5 감성 라벨과 점수를 보여줌. `inference/` 의 `config.json` 과 가중치를 그대로 불러 쓰므로 가중치를 따로 두지 않음.

```bash
pip install -r inference/requirements.txt gradio
python gradio/app_gradio.py
```

기본으로 `0.0.0.0:7860` 에 뜨고 `share=True` 라 공개 gradio.live 링크도 함께 생성됨(최대 7일). 공개 링크가 필요 없으면 `app_gradio.py` 맨 아래 `demo.launch(...)` 의 `share` 를 `False` 로 바꾸면 됨.

## 라이선스

사용하는 사전학습 모델과 데이터셋은 각 출처의 라이선스 및 이용약관을 따름.
