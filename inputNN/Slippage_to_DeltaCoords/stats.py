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
        # Если в названии папки есть "type__brown", заменяем его и всё после него на "surf"
        if "type__brown" in folder_name:
            folder_name = folder_name.split("type__brown")[0] + "surf"
            
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
    
    merged = r2_data.merge(mae_data, on="Feature")
    merged = merged.sort_values(by="R2", ascending=False).reset_index(drop=True)
    
    if merged.empty:
        r2_data = r2_data.sort_values(by="R2", ascending=False).reset_index(drop=True)
        mae_data = mae_data.set_index("Feature").reindex(r2_data["Feature"]).reset_index()
        merged = pd.DataFrame({
            "Feature": r2_data["Feature"],
            "R2": r2_data["R2"],
            "MAE": mae_data["MAE"]
        })

    x = np.arange(len(merged["Feature"]))
    feature_numbers = [str(i+1) for i in x]
    
    legend_labels = [f"{i+1}: {feat}" for i, feat in enumerate(merged["Feature"])]
    legend_text = "\n".join(legend_labels)
    
    fig, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Метрики для {target}", fontsize=14, fontweight="bold")
    
    ax_r2.bar(x, merged["R2"], color="royalblue", edgecolor="black", alpha=0.8)
    ax_r2.set_ylabel("R2", fontsize=11)
    ax_r2.set_title("Метрика R2", fontsize=12)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(feature_numbers, fontsize=10)
    ax_r2.grid(axis="y", linestyle="--", alpha=0.5)
    ax_r2.set_xlabel("Номер фичи", fontsize=11)
    
    ax_mae.bar(x, merged["MAE"], color="crimson", edgecolor="black", alpha=0.8)
    ax_mae.set_ylabel("MAE", fontsize=11)
    ax_mae.set_title("Метрика MAE", fontsize=12)
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(feature_numbers, fontsize=10)
    ax_mae.grid(axis="y", linestyle="--", alpha=0.5)
    ax_mae.set_xlabel("Номер фичи", fontsize=11)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax_mae.text(1.05, 1.0, legend_text, transform=ax_mae.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(r"C:\Users\User\Documents\MyPythonProjects\inputNN\Slippage_to_DeltaCoords\\" + f"{target}_graph.png",
                 dpi = 300,
                   bbox_inches = "tight")
