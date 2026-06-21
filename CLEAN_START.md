# Clean-start package

Gói này được tạo để train lại từ đầu trên Kaggle.

Không bao gồm:

- checkpoint `.pt`, `.bin`, `.safetensors`;
- thư mục output/tuning cũ;
- `wandb`, `runs`, `logs`;
- `__pycache__`;
- notebook cũ có execution output/training history.

Notebook `kaggle_ablation_fresh.ipynb` sẽ xóa các thư mục output trong `/kaggle/working` trước khi train, nên mỗi lần chạy là một clean run mới.
