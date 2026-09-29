"""이미지-텍스트 융합 모델 학습 스크립트.

텍스트 인코더(한국어 LM)를 end-to-end fine-tune 하고, 동결한 DINOv3 이미지 특징과
2토큰 self-attention 으로 융합해 22개 감성 라벨 중 5개를 예측한다.

설계
  - 텍스트가 이 과제의 지배적인 신호이므로, 이미지 branch 가 텍스트를 희석시키지
    않도록 projection + 경량 attention 으로만 관여시킨다.
  - discriminative LR 을 쓴다. 사전학습 backbone 은 낮게, 새로 만든 레이어는 높게.
  - DINOv3 특징은 dinov3_features/features_*.pt 캐시를 재사용한다. split 과 seed 가
    같으면 id 가 그대로 일치한다.

사용법 (최종 구성 5개 모델, description-only. inference/config.json 과 같은 조합)
  python train_fusion_final.py --tag klue_descOnly_cfgB --model klue/roberta-large --desc_only --lr 1e-5 --dropout 0.1 --bs 4 --grad_accum 4 --max_length 384 --epochs 15 --gpu 0
  python train_fusion_final.py --tag xlmr_descOnly      --model xlm-roberta-large --desc_only                                                              --gpu 1
  python train_fusion_final.py --tag kcbert_descOnly    --model beomi/kcbert-large --desc_only --max_length 300                                            --gpu 2
  python train_fusion_final.py --tag mbert_descOnly     --model bert-base-multilingual-cased --desc_only                                                   --gpu 3
  python train_fusion_final.py --tag kobigbird_descOnly --model monologg/kobigbird-bert-base --desc_only                                                   --gpu 4

--desc_only 를 빼면 description 에 구조적 메타데이터를 붙여 쓴다. 최종 구성과 다르므로
권장하지 않는다.

각 실행은 features/fusion_{tag}_val.npz 에 val_prob / val_ids / val_f1 을 저장한다.
다섯 개가 모두 생성되면 ensemble.py 로 가중치를 탐색한다.
"""
import argparse
import os
import random as _random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402

HERE = Path(__file__).resolve().parent
IMG_FEAT_DIR = HERE / "dinov3_features"
OUT_DIR = HERE / "features"
OUT_DIR.mkdir(exist_ok=True)


def seed_everything(s):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(s)
    _random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        pt = probs_pos * targets + probs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        modulator = (1 - pt) ** gamma
        return -(modulator * (loss_pos + loss_neg)).sum(dim=1).mean()


class ImageTextFusion(nn.Module):
    """klue-roberta(fine-tune, mean-pool) + DINOv3(frozen, pooled) -> 2-token self-attn -> 분류."""

    def __init__(self, text_model_name, image_dim, n_labels, hidden_dim=1024, num_heads=8, dropout=0.2):
        super().__init__()
        self.text_backbone = AutoModel.from_pretrained(text_model_name, use_safetensors=True)
        text_dim = self.text_backbone.config.hidden_size

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, n_labels)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        hid = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hid * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, image_feat):
        text_pooled = self.encode_text(input_ids, attention_mask)
        txt_tok = self.text_proj(text_pooled).unsqueeze(1)
        img_tok = self.image_proj(image_feat).unsqueeze(1)
        seq = torch.cat([txt_tok, img_tok], dim=1)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.ln(seq + attn_out)
        flat = self.drop(seq.flatten(1))
        return self.head(flat)


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, img, Y):
        self.enc, self.img, self.Y = enc, img, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return (self.enc["input_ids"][i], self.enc["attention_mask"][i],
                self.img[i], torch.tensor(self.Y[i], dtype=torch.float32))


def load_image_feats(split_name, ids):
    cache = torch.load(IMG_FEAT_DIR / f"features_{split_name}.pt", map_location="cpu", weights_only=False)
    id_to_idx = {id_: i for i, id_ in enumerate(cache["ids"])}
    order = [id_to_idx[i] for i in ids]
    return cache["img_features"][order].float()


def run(a):
    seed_everything(a.seed)
    device = f"cuda:{a.gpu}"

    recs = D.load_records()
    vocab = D.build_vocab(recs)
    train, va = D.split_records(recs)
    print(f"train={len(train)} val={len(va)} vocab={len(vocab)}")

    train_img = load_image_feats("train", [r["id"] for r in train])
    val_img = load_image_feats("val", [r["id"] for r in va])
    image_dim = train_img.shape[1]
    print(f"image_dim={image_dim}")

    use_fast = "deberta" not in a.model.lower()
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=use_fast)

    # 텍스트 입력은 description(annotations) 만 쓴다. --desc_only 주면 구조적 메타데이터
    # (문양유형/재질/시대 등)도 빼고 진짜로 description 필드 텍스트만 쓴다.
    def encode(recs_):
        if a.desc_only:
            docs = [D.redact(r["description"], vocab) if not a.keep_leak else r["description"] for r in recs_]
        else:
            docs = [D.build_doc(r, vocab, redact_emotions=not a.keep_leak) for r in recs_]
        return tok(docs, padding="max_length", truncation=True, max_length=a.max_length, return_tensors="pt")

    enc_tr = encode(train)
    enc_va = encode(va)
    Ytr = D.multihot(train, vocab)
    Yva = D.multihot(va, vocab)

    model = ImageTextFusion(a.model, image_dim, len(vocab), hidden_dim=a.hidden_dim,
                             num_heads=a.num_heads, dropout=a.dropout).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("text_backbone") and not any(nd in n for nd in no_decay)]
    backbone_params_nd = [p for n, p in model.named_parameters() if n.startswith("text_backbone") and any(nd in n for nd in no_decay)]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("text_backbone")]

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": a.lr, "weight_decay": a.wd},
        {"params": backbone_params_nd, "lr": a.lr, "weight_decay": 0.0},
        {"params": head_params, "lr": a.head_lr, "weight_decay": a.wd},
    ])
    micro_steps_per_epoch = len(train) // a.bs + 1
    opt_steps_per_epoch = -(-micro_steps_per_epoch // a.grad_accum)  # ceil
    steps = a.epochs * opt_steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, a.lr, a.head_lr], total_steps=steps, pct_start=0.1,
    )

    train_ds = DS(enc_tr, train_img, Ytr)
    if a.no_sampler:
        loader = torch.utils.data.DataLoader(train_ds, batch_size=a.bs, shuffle=True, num_workers=2)
    else:
        sample_weights = class_balanced_weights(train, vocab)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=a.bs, sampler=sampler, num_workers=2)

    crit = AsymmetricLoss()
    val_img_dev = val_img.to(device)
    val_ids_ordered = [r["id"] for r in va]

    best_f1 = -1.0
    best_val_prob = None
    for ep in range(1, a.epochs + 1):
        model.train()
        tot_loss = 0.0
        opt.zero_grad()
        for step, (input_ids, attn_mask, img_feat, y) in enumerate(loader):
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            img_feat, y = img_feat.to(device), y.to(device)
            logits = model(input_ids, attn_mask, img_feat)
            loss = crit(logits, y) / a.grad_accum
            loss.backward()
            if (step + 1) % a.grad_accum == 0 or (step + 1) == micro_steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            tot_loss += loss.item() * a.grad_accum

        model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(va), a.bs):
                input_ids = enc_va["input_ids"][i:i + a.bs].to(device)
                attn_mask = enc_va["attention_mask"][i:i + a.bs].to(device)
                img_feat = val_img_dev[i:i + a.bs]
                logits = model(input_ids, attn_mask, img_feat)
                all_logits.append(logits.cpu())
        val_logits = torch.cat(all_logits)
        val_prob = torch.sigmoid(val_logits).numpy()
        f1 = D.top5_f1(val_prob, va, vocab)

        print(f"[{a.tag}] ep{ep}/{a.epochs} loss={tot_loss/len(loader):.3f} F1@5={f1:.4f} (best {best_f1:.4f})", flush=True)

        if f1 > best_f1:
            best_f1 = f1
            best_val_prob = val_prob
            torch.save(model.state_dict(), OUT_DIR / f"fusion_{a.tag}_best.pt")

    np.savez(OUT_DIR / f"fusion_{a.tag}_val.npz", val_prob=best_val_prob,
             val_ids=np.array(val_ids_ordered), val_f1=best_f1, vocab=np.array(vocab))
    print(f"SAVED fusion_{a.tag}_val.npz  best F1@5={best_f1:.4f}")


def class_balanced_weights(records, vocab):
    l2i = {l: i for i, l in enumerate(vocab)}
    freq = np.zeros(len(vocab))
    for r in records:
        for e in r["emotions"]:
            freq[l2i[e]] += 1
    inv = 1.0 / np.clip(freq, 1, None)
    weights = []
    for r in records:
        w = max(inv[l2i[e]] for e in r["emotions"])
        weights.append(w)
    return torch.tensor(weights, dtype=torch.double)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="klue/roberta-large")
    p.add_argument("--tag", default="klue")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_leak", action="store_true", default=True)
    p.add_argument("--no_sampler", action="store_true", default=True)
    p.add_argument("--desc_only", action="store_true", default=False,
                    help="구조적 메타데이터 없이 description 필드 텍스트만 사용")
    a = p.parse_args()
    run(a)
