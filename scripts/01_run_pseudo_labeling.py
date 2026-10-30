import argparse
import os
import pandas as pd
from src.pseudo_labeling import (
    LABEL_COLS,
    KneeMRILabelExtractor,
    select_cases_ilp,
)
from src.utils import DriveCheckpointManager


def parse_args():
    parser = argparse.ArgumentParser(description="Chạy pipeline trích xuất pseudo-label từ báo cáo MRI bằng LLM.")
    parser.add_argument("--train_csv", type=str, default="data/raw/train.csv", help="Đường dẫn file train.csv gốc.")
    parser.add_argument("--checkpoint_csv", type=str, default="predictions_by_uid.csv", help="Tên file lưu checkpoint dự đoán.")
    parser.add_argument("--output_csv", type=str, default="data/processed/knee_mri_full_labeled_dataset.csv", help="Đường dẫn file kết quả đã gộp.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tên hoặc đường dẫn HuggingFace model.")
    parser.add_argument("--batch_size", type=int, default=2, help="Kích thước batch inference.")
    parser.add_argument("--k_fewshots", type=int, default=6, help="Số ca Few-shot tối ưu chọn qua MILP.")
    parser.add_argument("--use_drive", action="store_true", help="Kích hoạt đồng bộ checkpoint với Google Drive.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    drive_mgr = None
    if args.use_drive:
        try:
            drive_mgr = DriveCheckpointManager()
        except Exception as e:
            print(f"[Drive] Không thể kết nối Drive: {e}. Chuyển sang lưu cục bộ.")

    # 1. Đọc dữ liệu
    df_train = pd.read_csv(args.train_csv)
    if "StudyInstanceUID" not in df_train.columns and df_train.index.name == "StudyInstanceUID":
        df_train = df_train.reset_index()

    df_labeled = df_train[df_train["ACL"].notna()].copy()
    df_unlabeled = df_train[df_train["ACL"].isna()].copy()
    print(f"--> Ca đã có nhãn chuẩn: {len(df_labeled)} | Ca cần gán nhãn: {len(df_unlabeled)}")

    # 2. Khôi phục checkpoint
    id_cols = ["StudyInstanceUID"] + LABEL_COLS
    if args.use_drive and drive_mgr is not None:
        drive_mgr.download_checkpoint(args.checkpoint_csv, args.checkpoint_csv)

    processed_uids = set()
    if os.path.exists(args.checkpoint_csv) and os.path.getsize(args.checkpoint_csv) > 0:
        try:
            df_done = pd.read_csv(args.checkpoint_csv, dtype={"StudyInstanceUID": str})
            processed_uids = set(df_done["StudyInstanceUID"].dropna().astype(str).str.strip().tolist())
            print(f"--> [Resume] Đã tìm thấy {len(processed_uids)} ca đã xử lý trước đó.")
        except Exception as e:
            print(f"[Cảnh báo] Lỗi đọc checkpoint: {e}")
    else:
        pd.DataFrame(columns=id_cols).to_csv(args.checkpoint_csv, index=False)

    df_to_run = df_unlabeled[~df_unlabeled["StudyInstanceUID"].astype(str).isin(processed_uids)].reset_index(drop=True)
    print(f"--> Số ca còn lại cần xử lý: {len(df_to_run)}")

    # 3. Chọn ca Few-shot tối ưu qua MILP
    print("--> Chọn tập ca Few-Shot tối ưu bằng 0-1 MILP...")
    df_fewshots = select_cases_ilp(df_labeled, LABEL_COLS, k=args.k_fewshots, min_pos_per_label=1)

    # 4. Khởi tạo LLM Extractor
    extractor = KneeMRILabelExtractor(model_id=args.model_id)

    # 5. Tiến hành Inference tuần tự theo batch và đồng bộ
    total = len(df_to_run)
    with open(args.checkpoint_csv, mode="a", encoding="utf-8") as f_out:
        for i in range(0, total, args.batch_size):
            batch_df = df_to_run.iloc[i : i + args.batch_size]
            uids = batch_df["StudyInstanceUID"].astype(str).tolist()
            reports = batch_df["Report"].astype(str).tolist()

            preds = extractor.extract_batch(reports, few_shots_df=df_fewshots, batch_size=args.batch_size)

            for uid, pred in zip(uids, preds):
                row = [uid] + [str(int(pred.get(c, 0))) for c in LABEL_COLS]
                f_out.write(",".join(row) + "\n")
            f_out.flush()

            if args.use_drive and drive_mgr is not None and ((i // args.batch_size) + 1) % 5 == 0:
                drive_mgr.update_checkpoint({}, args.checkpoint_csv, keep_local=True, keep_revision=False)

    if args.use_drive and drive_mgr is not None:
        drive_mgr.update_checkpoint({}, args.checkpoint_csv, keep_local=True, keep_revision=True)

    # 6. Hợp nhất tập dữ liệu hoàn chỉnh
    df_preds = pd.read_csv(args.checkpoint_csv, dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID")
    df_merged = df_train.merge(df_preds, on="StudyInstanceUID", how="left", suffixes=("_orig", ""))

    for c in LABEL_COLS:
        if f"{c}_orig" in df_merged.columns:
            df_merged[c] = df_merged[c].combine_first(df_merged[f"{c}_orig"])
            df_merged.drop(columns=[f"{c}_orig"], inplace=True)

    df_merged.to_csv(args.output_csv, index=False)
    print(f"--> [Hoàn tất] File gán nhãn đầy đủ được lưu tại: {args.output_csv} ({len(df_merged)} ca)")


if __name__ == "__main__":
    main()