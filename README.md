# 전통문양 감성 분류 - 이미지 + 텍스트 융합 앙상블

전통문양 이미지에 감성 형용사를 다는 태깅 모델입니다. 22개 어휘에 대한 multi-label classification 이고, 흑백 문양 이미지 한 장과 그 이미지에 대한 설명 문장을 받아 형용사 5개를 출력합니다. 정답 형용사도 항상 5개이고 전문가가 붙인 것입니다. 정답과 예측이 모두 5개이므로 precision = recall = F1 이고, 지표는 F1@5 입니다.

```
F1@5 = |예측 ∩ 정답| / 5
```

이 저장소는 이미지 인코더 하나와 텍스트 인코더 다섯 개를 각각 융합한 모델 5종을 학습하고, 그 확률을 가중합해 앙상블하는 코드입니다.

| | |
|---|---|
| val F1@5 | **0.8000** (446장) |
| 이미지 인코더 | DINOv3 (동결) |
| 텍스트 인코더 | 한국어·다국어 언어모델 5종 (fine-tune) |
| 융합 | 2토큰 self-attention |
| 앙상블 | 5개 모델 확률의 Dirichlet 랜덤서치 가중합 |

설계 근거와 실험 기록은 저장소가 아니라 프로젝트 노트에 있습니다.
→ [전통문양 감성 라벨 예측](https://chb2066.github.io/projects/traditional-patterns/)

## 개요

**이미지 - DINOv3 (동결)**

ConvNeXtV2, SigLIP2, FG-CLIP2, DINOv3 를 비교했을 때 이미지 단독 성능이 가장 좋았습니다. 학습 중 가중치를 갱신하지 않고 이미지 feature 를 뽑는 역할로만 씁니다. feature 는 미리 뽑아 두므로 학습 중에 이미지 픽셀을 다시 읽지 않습니다.

**텍스트 - 언어모델 5종 (fine-tune)**

입력은 설명 문장 하나뿐입니다. 문양유형·재질·시대 같은 구조적 메타데이터는 넣지 않습니다.

| 모델 | HuggingFace 이름 | 단독 F1@5 |
|---|---|---:|
| kobigbird | `monologg/kobigbird-bert-base` | 0.7861 |
| klue | `klue/roberta-large` | 0.7794 |
| xlmr | `xlm-roberta-large` | 0.7753 |
| mbert | `bert-base-multilingual-cased` | 0.7753 |
| kcbert | `beomi/kcbert-large` | 0.7682 |

**융합 - 2토큰 self-attention**

이미지 벡터와 텍스트 벡터를 같은 1024차원으로 맞추고, 둘을 토큰 2개짜리 시퀀스로 취급해 self-attention(Q=K=V)을 한 번 적용합니다. 그 출력으로 22개 라벨 점수를 내고 상위 5개를 고릅니다.

**앙상블**

5개 모델의 확률을 가중합합니다. 가중치는 val 446장 기준 Dirichlet 랜덤서치로 탐색한 값이며 `inference/config.json` 의 `ensemble_weight` 에 들어 있습니다.

## 설치

```bash
pip install -r inference/requirements.txt
```

HuggingFace 백본은 최초 실행 시 자동으로 내려받아 캐시합니다(총 5~6GB, 인터넷 연결 필요). GPU 없이도 동작하지만 텍스트 백본 5개를 CPU로 돌리면 많이 느립니다.

## 데이터

ETRI 과제로 구축된 데이터셋이라 이미지도 어노테이션도 공개할 수 없습니다. 따라서 표기된 성능을 그대로 재현할 수는 없습니다.

다만 텍스트 입력으로 쓰는 `description` 은 문화포털이 원래 제공하는 설명 데이터이므로, 문화포털 등에 공개된 전통문양 이미지와 그 설명을 그대로 넣어 추론해 볼 수 있습니다.

학습 코드가 기대하는 레코드 구성은 다음과 같습니다.

| 필드 | 설명 |
|---|---|
| 이미지 | 흑백 선화 문양 이미지 파일 |
| `description` | 문화포털이 원래 제공하는 이미지 설명 데이터. 학습에 쓰는 유일한 텍스트 입력 |
| 감성 라벨 | 22개 어휘 중 5개. 전문가가 붙인 정답 태그 |

데이터 위치는 환경변수로 지정합니다.

```bash
export PATTERN_DATA_ROOT=/path/to/dataset
```

`dataset.py` 의 `load_records()` 가 split 을 만들 때 이미지 파일의 존재 여부를 확인하므로, 동일한 3,954 / 446 split 을 재현하려면 원본 파일 자체는 있어야 합니다.

## 사용법

### 학습

```bash
python train/train_fusion_final.py --desc_only    # 모델 1개 학습
python train/ensemble.py                          # 5개 가중치 탐색
python val/analyze_val.py                         # val 결과 분석
```

| 옵션 | 설명 |
|---|---|
| `--desc_only` | 설명 문장만 사용. 붙이지 않으면 구조적 메타데이터를 함께 넣습니다(비권장, 최종 구성과 다름) |

### 추론

데이터셋 없이 가중치만 있으면 됩니다.

```bash
cd inference
python infer.py --image <이미지경로> --description "<설명 문장>"
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--image` | 입력 이미지 경로 | 필수 |
| `--description` | 설명 문장. 메타데이터 없이 순수 서술만 | 필수 |
| `--device` | `cuda` / `cpu` | 자동 |

출력 예시

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

### 데모

이미지와 `description` 을 넣으면 top-5 감성 라벨과 점수를 보여줍니다. `inference/` 의 `config.json` 과 가중치를 그대로 불러 쓰므로 가중치를 따로 두지 않습니다.

```bash
pip install -r inference/requirements.txt gradio
python gradio/app_gradio.py
```

기본으로 `0.0.0.0:7860` 에 뜨고 `share=True` 라 공개 gradio.live 링크도 함께 생성됩니다(최대 7일). 공개 링크가 필요 없으면 `app_gradio.py` 맨 아래 `demo.launch(...)` 의 `share` 를 `False` 로 바꾸세요.

## 저장소 구조

```
train/
  train_fusion_final.py   학습 (--desc_only 로 설명 문장만 사용)
  dataset.py              레코드 로딩, split, top5 F1
  ensemble.py             5개 모델 가중치 탐색
inference/
  infer.py                배포용 추론. 이 폴더 밖 코드에 의존하지 않음
  config.json             모델 구성과 앙상블 가중치
  docs/MODEL.md           모델 구조 상세
  docs/fusion_architecture.html   구조도
val/
  analyze_val.py          라벨별 성능, 혼동 분석, 모델 간 예측 다양성
gradio/
  app_gradio.py           데모
  collect_description.py  새 이미지의 description 을 받아 모으는 별도 도구 (추론 없음)
```

모델 가중치 5개는 파일 하나당 GitHub 한도 100MB 를 넘어 포함하지 않았습니다.

## 재현성

시드 42 로 고정했습니다. 데이터 셔플, 모델 초기화, dropout 등 무작위성이 들어가는 지점에 전부 적용했고, 같은 데이터가 있으면 `ensemble.py` 재실행 시 0.8000 이 그대로 나오는 것을 확인했습니다.

## 라이선스

사용하는 사전학습 모델과 데이터셋은 각 출처의 라이선스 및 이용약관을 따릅니다.
