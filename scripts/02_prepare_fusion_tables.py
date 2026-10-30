import argparse
import os
from functools import reduce
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Tạo các bảng tổng hợp đơn mặt phẳng và study fusion metadata.")
    parser.add_argument("--train_csv", type=str, default="data/processed/knee_mri_full_labeled_dataset.csv", help="File dữ liệu đã đủ 12 nhãn.")
    parser.add_argument("--train_series_csv", type=str, default="data/raw/train_series.csv", help="File series chụp của train.")
    parser.add_argument("--test_series_csv", type=str, default="data/raw/test_series.csv", help="File series chụp của test.")
    parser.add_argument("--sample_sub_csv", type=str, default="data/raw/sample_submission.csv", help="File submission mẫu để lấy study UID test.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Thư mục lưu các file CSV đầu ra.")
    return parser.parse_args()


def build_fusion_dataframe(series_df: pd.DataFrame, planes: list, labels_df: pd.DataFrame = None, is_test: bool = False) -> pd.DataFrame:
    s_df = series_df.copy()
    s_df["priority"] = s_df["Fat_Suppression"].fillna(0).astype(int) if "Fat_Suppression" in s_df.columns else 0
    s_df = s_df.sort_values(by=["StudyInstanceUID", "priority"], ascending=[True, False])

    plane_dfs = []
    for plane in planes:
        sub = s_df[s_df["Anatomical_Plane"] == plane].drop_duplicates(subset=["StudyInstanceUID"])
        key = plane.lower()[:3]
        sub = sub[["StudyInstanceUID", "SeriesInstanceUID"]].rename(
            columns={"SeriesInstanceUID": f"SeriesInstanceUID_{key}"}
        )
        plane_dfs.append(sub)

    fusion_df = reduce(lambda left, right: pd.merge(left, right, on="StudyInstanceUID", how="outer"), plane_dfs)

    if not is_test and labels_df is not None:
        fusion_df = pd.merge(fusion_df, labels_df, on="StudyInstanceUID", how="inner")

    return fusion_df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    planes = ["Sagittal", "Coronal", "Axial"]

    print("--> Nạp dữ liệu metadata...")
    train = pd.read_csv(args.train_csv)
    if "Report" in train.columns:
        train = train.drop(columns=["Report"])
    train_series = pd.read_csv(args.train_series_csv)

    # 1. Tạo các bảng đơn mặt phẳng (df_sagittal, df_coronal, df_axial)
    for plane in planes:
        sub = train_series[train_series["Anatomical_Plane"] == plane].copy()
        merged = pd.merge(sub, train, on="StudyInstanceUID", how="inner")
        out_path = os.path.join(args.output_dir, f"df_{plane.lower()}.csv")
        merged.to_csv(out_path, index=False)
        print(f"    - Đã xuất {out_path}: {len(merged)} dòng.")

    # 2. Tạo bảng Study Fusion tập Train
    df_study_fusion = build_fusion_dataframe(train_series, planes, labels_df=train, is_test=False)
    out_fusion = os.path.join(args.output_dir, "df_study_fusion.csv")
    df_study_fusion.to_csv(out_fusion, index=False)
    print(f"--> Đã xuất bảng train fusion: {out_fusion} ({len(df_study_fusion)} ca).")

    # 3. Tạo bảng Study Fusion tập Test
    if os.path.exists(args.test_series_csv):
        test_series = pd.read_csv(args.test_series_csv)
        if os.path.exists(args.sample_sub_csv):
            sub_sample = pd.read_csv(args.sample_sub_csv)
            valid_studies = sub_sample[["StudyInstanceUID"]].drop_duplicates()
        else:
            valid_studies = test_series[["StudyInstanceUID"]].drop_duplicates()

        test_subset = test_series[test_series["StudyInstanceUID"].isin(valid_studies["StudyInstanceUID"])].copy()
        df_test_raw = build_fusion_dataframe(test_subset, planes, is_test=True)
        df_test_fusion = pd.merge(valid_studies, df_test_raw, on="StudyInstanceUID", how="left")

        out_test = os.path.join(args.output_dir, "df_test_fusion.csv")
        df_test_fusion.to_csv(out_test, index=False)
        print(f"--> Đã xuất bảng test fusion: {out_test} ({len(df_test_fusion)} ca).")


if __name__ == "__main__":
    main()