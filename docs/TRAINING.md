# Quy trinh huan luyen ABSA

## Du lieu

`train.multiple.json` va `test.multiple.json` chua cac cau co mot hoac nhieu
aspect. Pipeline bung moi aspect thanh mot mau rieng, sau do:

1. Giu nguyen official test set.
2. Loai khoi train cac cau xuat hien trong test.
3. Chia validation theo nhom cau de cac aspect cua cung mot cau khong nam o hai
   partition khac nhau.
4. Chon hyperparameter va epoch chi bang validation Macro-F1.
5. Khoi tao lai mo hinh, refit tren toan bo train sach va chi danh gia test sau
   khi da chot cau hinh.

## Mo hinh

Input la cap `sentence + aspect`. Encoder Transformer nhan ca cau va target
aspect, sau do classification head du doan mot trong ba nhan.

Pipeline ho tro ba cach tao representation:

- `cls`: dung token dau tien cua encoder. Day la cau hinh mac dinh vi da co ket
  qua thuc nghiem tot hon tren du lieu hien tai.
- `aspect_attention`: pool token aspect va dung aspect lam query attention tren
  cau.
- `hybrid`: ket hop CLS, aspect representation va attentive context.

Cau hinh mac dinh:

- encoder: `microsoft/deberta-v3-base`;
- pooling: `cls`;
- encoder/head learning rate: `1e-5`;
- dropout: `0.2`;
- class weighting: `none`;
- label smoothing: `0.0`;
- scheduler: `none`;
- max length: `128`;
- gradient clipping: `1.0`.

Day la diem khoi dau da co ket qua tot trong cac run truoc, khong phai ket luan
rang day la optimum cho moi seed.

## Tuning sach

Vong tuning dau cho thay CLS tot hon aspect attention/hybrid, nhung cau hinh
thang tren mot split chi dat 77.42% Macro-F1 tren test seed 2024. Vong hai vi
vay thu sau bien the quanh CLS va danh gia tren ba cap split/model seed.

```bash
python tune.py \
  --dataset rest \
  --search-space refined \
  --validation-seed-pairs 2024:2024,42:42,3407:3407 \
  --final-seeds 2024,42,3407 \
  --amp \
  --run-final
```

Sau moi run, score robust duoc tinh bang `mean Macro-F1 - std`. Official test
chi duoc danh gia trong cac final run cua cau hinh thang. Checkpoint va ZIP lon
cua trial duoc xoa tu dong; CSV, JSON va bieu do validation van duoc giu lai.

Thay doi ky thuat v3:

- Refit giu nguyen scheduler horizon cua giai doan chon epoch. Neu epoch 9 duoc
  chon trong lich 15 epoch, refit dung 9 epoch dau cua cung lich 15 epoch.
- Bias va LayerNorm khong bi weight decay.
- Ho tro label smoothing, sqrt class weighting, focal loss va mean-last-4.
- Temperature scaling duoc fit tren validation va ap dung khi bao cao test.
- Probability duoc tinh float32 va normalize float64 truoc metric.

De thu nhanh mot cau hinh:

```bash
python tune.py --trials cls_reference --tuning-seeds 2024 --amp
```

## Danh gia

JSON ket qua luu:

- Accuracy va Balanced Accuracy;
- Precision, Recall, F1 theo macro, micro, weighted va tung lop;
- confusion matrix;
- cross-entropy loss va log loss;
- ROC-AUC macro, micro, weighted va tung lop;
- confidence trung binh cho du doan dung/sai;
- Brier score va Expected Calibration Error;
- metric rieng cho aspect da xuat hien va chua xuat hien trong train.

`predictions.csv` luu day du thong tin de phan tich thu cong cac mau dung va sai.

## Chay truc tiep

```bash
python train.py --dataset rest --amp --refit-full-train
```

Lenh nay dung cau hinh mac dinh, chon epoch bang validation, refit full train va
danh gia test. De bao cao ket qua cuoi, uu tien chay `tune.py --run-final`.
