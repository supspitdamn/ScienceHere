import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_folder = r"C:\Users\User\Documents\MyPythonProjects\inputNN\Slippage_to_DeltaCoords"
links_to_csv = []
feature_names = []

for root, folders, files in os.walk(root_folder):
    if "FINAL_metrics_test.csv" in files:
        links_to_csv.append(os.path.join(root, "FINAL_metrics_test.csv"))
        folder_name = os.path.basename(root)
        feature_names.append(folder_name)

all_data = []

for folder, link in zip(feature_names, links_to_csv):
    df_metrics = pd.read_csv(link, encoding="utf-8-sig")
    metric_col_name = df_metrics.columns[0]
    df_metrics[metric_col_name] = df_metrics[metric_col_name].astype(str).str.strip()
    to_remove = ["MAPE, %", "MSE"]
    df_metrics = df_metrics[~df_metrics[metric_col_name].isin(to_remove)]
    df_metrics["Feature"] = folder
    all_data.append(df_metrics)

final_df = pd.concat(all_data, ignore_index=True)
metric_col_name = final_df.columns[0]

final_df[metric_col_name] = final_df[metric_col_name].astype(str).str.strip().str.upper()

df_r2_all = final_df[final_df[metric_col_name] == "R2"].copy()
df_mae_all = final_df[final_df[metric_col_name] == "MAE"].copy()

targets = ["Дельта Х", "Дельта У", "Дельта Фи"]

for target in targets:
    r2_data = df_r2_all[["Feature", target]].rename(columns={target: "R2"})
    mae_data = df_mae_all[["Feature", target]].rename(columns={target: "MAE"})
    
    r2_data["Feature"] = r2_data["Feature"].astype(str).str.strip()
    mae_data["Feature"] = mae_data["Feature"].astype(str).str.strip()
    
    r2_data["R2"] = pd.to_numeric(r2_data["R2"], errors='coerce')
    mae_data["MAE"] = pd.to_numeric(mae_data["MAE"], errors='coerce')
    
    merged = r2_data.merge(mae_data, on="Feature", how="outer")
    merged["R2"] = merged["R2"].fillna(0)
    merged["MAE"] = merged["MAE"].fillna(0)
    merged = merged.sort_values(by="R2", ascending=False).reset_index(drop=True)

    y_pos = np.arange(len(merged["Feature"]))
    fig_height = max(6, len(merged["Feature"]) * 0.4)
    
    fig_r2, ax_r2 = plt.subplots(figsize=(10, fig_height))
    ax_r2.barh(y_pos, merged["R2"], color="royalblue", edgecolor="black", alpha=0.8)
    ax_r2.set_xlabel("R2", fontsize=11, fontweight="bold")
    ax_r2.set_title(f"Метрика R2 для {target}", fontsize=13, fontweight="bold", pad=15)
    ax_r2.set_yticks(y_pos)
    ax_r2.set_yticklabels(merged["Feature"], fontsize=10)
    ax_r2.invert_yaxis()
    ax_r2.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, f"{target}_R2.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_r2)
    
    fig_mae, ax_mae = plt.subplots(figsize=(10, fig_height))
    ax_mae.barh(y_pos, merged["MAE"], color="crimson", edgecolor="black", alpha=0.8)
    ax_mae.set_xlabel("MAE", fontsize=11, fontweight="bold")
    ax_mae.set_title(f"Метрика MAE для {target}", fontsize=13, fontweight="bold", pad=15)
    ax_mae.set_yticks(y_pos)
    ax_mae.set_yticklabels(merged["Feature"], fontsize=10)
    ax_mae.invert_yaxis()
    ax_mae.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_folder, f"{target}_MAE.png"), dpi=300, bbox_inches="tight")
    plt.close(fig_mae)