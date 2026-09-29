# 최종 모델 상세 설명 - 이미지+텍스트 융합 앙상블 (val F1@5 = 0.8000)

구조도: [`fusion_architecture.html`](./fusion_architecture.html)

---

## 1. 학습에 사용된 모델

### 이미지 인코더 (1개, 공통)
| 항목 | 내용 |
|---|---|
| 모델 | `facebook/dinov3-vitl16-pretrain-lvd1689m` (DINOv3, ViT-Large) |
| 학습 여부 | **완전 고정(frozen)** - 이번 학습에서 가중치 전혀 안 바뀜 |
| 출력 | CLS 토큰 + patch 토큰 평균을 concat한 pooled 벡터 (1024×2 = **2048차원**) |
| 선택 이유 | 이 프로젝트에서 CLIP/SigLIP2/FG-CLIP2/DINOv3를 이미지 단독으로 비교했을 때 DINOv3가 가장 성능이 좋았음(self-supervised라 텍스트 정렬에 편향되지 않은 순수 시각 표현). 5개 텍스트 모델 전부가 이 하나의 이미지 인코더를 공유함. |

### 텍스트 인코더 (5개, 각각 fine-tuning, description 텍스트만 사용)
| 태그 | 모델 | 사전학습 특징 | 최종 개별 성능(F1@5) |
|---|---|---|---|
| `klue` | klue/roberta-large | 한국어 뉴스·위키 코퍼스, RoBERTa 구조 | 0.7794 (fine-tuning 설정 튜닝) |
| `xlmr` | xlm-roberta-large | 100개 언어 다국어 코퍼스 | 0.7753 |
| `kcbert` | beomi/kcbert-large | 한국어 **커뮤니티/구어체** 텍스트(네이버 뉴스 댓글) | 0.7682 |
| `mbert` | bert-base-multilingual-cased | 104개 언어 다국어 코퍼스, mBERT 구조(xlmr과는 세대/토크나이저가 다름) | 0.7753 |
| `kobigbird` | monologg/kobigbird-bert-base | 한국어 코퍼스, BigBird(sparse attention) 구조 | 0.7861 (단독 최고) |

5개를 고른 기준은 **서로 다른 사전학습 코퍼스 또는 다른 아키텍처**를 하나씩 확보하는 것
klue-roberta 하나만으로는 0.78 근처에서 정체되는데, 성격이 다른 모델을 섞으면 각자
실수하는 지점이 달라서(오류가 decorrelated) 앙상블 시 유의미하게 올라감.

> **텍스트 입력은 description 문장만 사용한다** (문양유형/재질/시대 같은 구조적 메타데이터는
> 넣지 않음). 이전 버전은 실수로 메타데이터를 같이 넣고 있었는데, description만 쓰는 게
> 원래 의도였다는 게 확인돼서 전부 다시 학습시켰음. 메타데이터를 빼도 개별 모델 성능
> 차이는 크지 않았음(±0.003~0.004, 노이즈 수준) - §4-3 참고.

> 원래는 klue_bert를 포함한 4개 조합(0.7991)이었는데, klue_bert 대신 kobigbird를 넣은
> 5개 조합이 0.8000으로 더 높아서 이걸 채택함 - §5-2 참고.

---

## 2. 융합(fusion) 구조

이미지 pooled 벡터(2048-d)와 텍스트 pooled 벡터(모델별 768~1024-d)를 각각 `Linear`로
**같은 1024차원**으로 projection한 뒤, 이 둘을 "토큰 2개짜리 시퀀스"로 보고
**8-head self-attention 1층**을 태운다. Q, K, V가 전부 이 2-토큰 시퀀스에서 나오는
표준 self-attention이라, 두 토큰(이미지/텍스트) 모두 [자기 자신, 상대방]을 한 번씩
참고한다. 그 결과(residual+LayerNorm)를 다시 이어붙여서(2048-d) 작은 MLP를 거쳐
22개 라벨 점수를 출력하고, sigmoid 확률 중 상위 5개를 최종 예측으로 고른다.

- 왜 attention을 쓰는가: 단순 concat보다 "이미지가 텍스트를, 텍스트가 이미지를 한 번
  참고하는" 여지를 준다. 사전 비교 실험에서 concat(0.661) < attention(0.677)로 확인함.
- 왜 무겁게(patch-token 단위) 안 하는가: 이 프로젝트에서 patch-token(256개) + 라벨별
  쿼리(22개) 방식의 ML-Decoder를 시도했을 때, 데이터가 ~4,000장뿐이라 학습이 제대로
  안 되고 출력이 collapse(로짓이 거의 균일해짐)하는 걸 이미 확인함. 그래서 토큰을
  **딱 2개**로 극단적으로 가볍게 만들어서 같은 문제를 피함.

---

## 3. 학습 디테일

| 항목 | 값 | 비고 |
|---|---|---|
| Loss | AsymmetricLoss (gamma_neg=4, gamma_pos=1, clip=0.05) | 쉬운 negative를 억제해서 클래스 불균형 완화 |
| 샘플러 | 사용 안 함 (natural 분포 그대로) | class-balanced sampler를 써봤는데 top-5 지표엔 오히려 손해라는 게 별도 실험에서 확인됨 |
| 옵티마이저 | AdamW, discriminative LR (backbone 낮게, head 높게) | |
| 스케줄러 | OneCycleLR (pct_start=0.1) | 웜업 후 코사인 감쇠 |
| Epoch | 12~15 (모델별 수렴 시점까지 소폭 튜닝) | |
| Batch size | 기본 16 (klue는 GPU 자원 부족으로 4, gradient accumulation 4로 실효 16 유지) | |
| Gradient clipping | max_norm=1.0 | |
| 텍스트 입력 | **description 문장만** (구조적 메타데이터 미포함) | 원래 설계 의도가 설명 문장만 쓰는 것이었다 |
| max_length | 모델별 300~512 (kcbert는 300 - 원 모델 특성상 짧게) | |
| 데이터 분할 | train 3,954 / val 446, iterative stratification (seed=42) | seed 고정이라 항상 같은 분할 |

---

설계 판단의 근거(왜 이미지는 얼리고 텍스트는 학습시켰는지, 왜 discriminative LR 을 썼는지,
왜 5개를 앙상블했는지)는 프로젝트 노트에 있다.
→ https://chb2066.github.io/projects/traditional-patterns/
