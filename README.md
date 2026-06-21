# Aspect-Based Sentiment Classification

Du an huan luyen mo hinh phan loai cam xuc theo khia canh (ABSA). Moi mau gom
mot cau, mot aspect trong cau va mot nhan `negative`, `neutral` hoac `positive`.

## Cai dat

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tren Windows PowerShell, kich hoat moi truong bang:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Huan luyen

Quy trinh khuyen nghi hien tai la repeated validation tren ba split/model seed,
sau do moi refit va danh gia test:

```bash
python tune.py \
  --dataset rest \
  --search-space refined \
  --validation-seed-pairs 2024:2024,42:42,3407:3407 \
  --final-seeds 2024,42,3407 \
  --amp \
  --run-final
```

De chay truc tiep cau hinh mac dinh da co ket qua tot trong cac run truoc:

```bash
python train.py \
  --dataset rest \
  --model microsoft/deberta-v3-base \
  --epochs 15 \
  --patience 4 \
  --batch-size 8 \
  --amp \
  --refit-full-train
```

Quy trinh giu nguyen official test set, loai cau train trung voi test va chia
validation theo toan bo cau. Checkpoint duoc chon bang validation Macro-F1, sau
do mo hinh duoc khoi tao lai va huan luyen tren toan bo train trong so epoch da
chon.

Khong co bo tham so nao duoc coi la toi uu truoc khi chay tuning. `tune.py` so
sanh cac learning rate, regularization va pooling strategy chi tren validation;
official test khong duoc dung de chon cau hinh.

Phien ban v3 con giu nguyen learning-rate horizon khi refit, loai weight decay
khoi bias/LayerNorm, ho tro focal loss, mean-last-4 pooling va temperature
scaling. Notebook [nlp-baseline.ipynb](nlp-baseline.ipynb) giu ket qua vong mot
va co san cell de chay vong hai.

## Ket qua

Moi lan chay tao mot bo artifact trong `outputs/`:

- checkpoint `.pt`;
- JSON chua cau hinh, thong ke du lieu, lich su train va toan bo metric;
- CSV metric tong hop va metric theo tung lop;
- CSV lich su tung epoch;
- CSV du doan gom cau, aspect, nhan that, nhan du doan, confidence va xac suat
  cua tung lop;
- learning curves;
- confusion matrix dang count va normalized;
- bieu do Precision, Recall, F1 theo tung lop;
- ROC curve va ROC-AUC one-vs-rest;
- confidence histogram va reliability diagram;
- file ZIP gom toan bo artifact.

Metric bao gom Accuracy, Balanced Accuracy, Precision, Recall, F1 theo macro,
micro, weighted va tung lop, loss, ROC-AUC, Brier score, Expected Calibration
Error va thong ke confidence.

Xem [docs/TRAINING.md](docs/TRAINING.md) de biet chi tiet va [KAGGLE.md](KAGGLE.md)
de chay tren Kaggle.
