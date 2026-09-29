# Korean Traditional Pattern Emotion Tagging

전통문양 이미지와 설명 문장을 받아 22개 감성 형용사 중 5개를 태깅함. DINOv3 이미지 feature 와 fine-tune 한 한국어 언어모델 5종을 2토큰 self-attention 으로 융합한 뒤 앙상블함.

| | |
|---|---|
| **val F1@5** | **0.8000** |
| Task | multi-label classification, top-5 of 22 |
| Split | train 3,954 / val 446 |
| Image encoder | DINOv3 ViT-L (frozen) |
| Text encoder | 한국어·다국어 LM 5종 (fine-tuned) |
| Fusion | 2-token self-attention |
| Ensemble | Dirichlet random search |

설계 근거와 실험 기록: [프로젝트 노트](https://chb2066.github.io/projects/traditional-patterns/)

## Results

**Image only**

| Approach | Model | F1@5 |
|---|---|---:|
| CNN | ConvNeXtV2-large | 0.560 |
| ViT | EVA-02-large, end-to-end | 0.571 |
| CLIP-style + cls head | SigLIP2 | 0.567 |
| CLIP-style + cls head | FG-CLIP2 | 0.576 |
| Foundation + cls head | DINOv3 | **0.579** |

다섯이 0.56~0.58 안에 몰림. 이미지 채널의 상한으로 판단함.

**Image + description**

| Approach | Model | F1@5 |
|---|---|---:|
| CLIP-style + cls head | SigLIP2 | 0.622 |
| CLIP-style + cls head | FG-CLIP2 | **0.623** |

description 을 더하면 오르지만 목표 0.80 에는 못 미침.

**Channel ablation**

| Channel | Method | F1@5 |
|---|---|---:|
| Description | klue/roberta-large, fine-tune | **0.774** |
| Metadata (9 fields) | multi-hot + linear probe | 0.561 |
| Image | DINOv3 | 0.579 |

description 단독이 이미지보다 약 20점 높음. description 외 메타데이터는 기여가 작음.

**Text encoder comparison**

| Encoder | Setting | F1@5 |
|---|---|---:|
| klue/roberta-large | fine-tune | **0.774** |
| FG-CLIP2 | text encoder only, fine-tune | 0.561 |
| FG-CLIP2 | frozen image + unfrozen text | 0.621 |

같은 텍스트를 넣어도 BERT 계열이 CLIP 계열 텍스트 타워를 크게 앞섬. 채널이 담을 수 있는 정보량보다 모델 구조가 지배적임.

**Final ensemble**

| Text encoder | F1@5 |
|---|---:|
| monologg/kobigbird-bert-base | 0.7861 |
| klue/roberta-large | 0.7794 |
| xlm-roberta-large | 0.7753 |
| bert-base-multilingual-cased | 0.7753 |
| beomi/kcbert-large | 0.7682 |
| **Ensemble (5)** | **0.8000** |

후보 6개의 부분집합 63개를 Dirichlet 가중 랜덤서치로 전부 탐색함. 4개 0.7991, 6개 0.7996, 최선의 5개가 0.8000.

## Architecture

```
image ──────▶ DINOv3 (frozen) ─────▶ proj ─▶ 1 token ─┐
                                                      ├─▶ self-attn ─▶ linear ─▶ 22
description ─▶ Korean LM (fine-tune) ▶ proj ─▶ 1 token ─┘
```

- 이미지 feature 는 미리 뽑아 두고 학습 중 픽셀을 다시 읽지 않음
- 텍스트 입력은 description 문장만. 구조적 메타데이터는 넣지 않음
- 모달리티당 pooled 벡터 하나씩, 토큰 2개 시퀀스에 self-attention 한 번
- backbone LR 을 낮게, head LR 을 높게 (discriminative LR)

## Installation

```bash
pip install -r inference/requirements.txt
```

HuggingFace 백본은 최초 실행 시 자동으로 받음 (약 5~6GB).

## Training

```bash
export PATTERN_DATA_ROOT=/path/to/dataset

python train/train_fusion_final.py --desc_only   # 단일 모델
python train/ensemble.py                         # 앙상블 가중치 탐색
python val/analyze_val.py                        # 라벨별 성능 분석
```

## Inference

```bash
cd inference
python infer.py --image <path> --description "<설명 문장>"
```

```python
from infer import EnsembleModel
model = EnsembleModel()
labels, scores = model.predict(image_path, description, topk=5)
```

```
Top-5 예측:
  고전적인: 0.9301
  조화로운: 0.9059
  소박한: 0.8168
  단순한: 0.7357
  우아한: 0.7333
```

## Demo

```bash
python gradio/app_gradio.py
```

## Repository

```
train/      train_fusion_final.py, dataset.py, ensemble.py
inference/  infer.py, config.json, docs/MODEL.md
val/        analyze_val.py
gradio/     app_gradio.py, collect_description.py
```

모델 가중치 5개는 파일당 GitHub 100MB 한도를 넘어 포함하지 않음.

## Data

ETRI 과제로 구축된 데이터셋이라 이미지도 어노테이션도 공개할 수 없음. 표기된 성능은 그대로 재현할 수 없음.

| Field | |
|---|---|
| image | 흑백 선화 문양 이미지 |
| `description` | 문화포털이 제공하는 이미지 설명 데이터 |
| labels | 22개 어휘 중 5개. 전문가가 부착 |

텍스트 입력이 문화포털 제공 데이터이므로, 문화포털에 공개된 전통문양 이미지와 설명으로 추론은 해볼 수 있음.

## License

사용한 사전학습 모델과 데이터셋은 각 출처의 라이선스 및 이용약관을 따름.
