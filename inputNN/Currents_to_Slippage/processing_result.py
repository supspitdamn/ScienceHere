import os
import matplotlib.pyplot as plt
import pandas as pd

root_dir = r"./Currents_to_Slippage/Seeking_for_best_features_05-05-2026_18-18-28"
average_results = []

for root, dirs, files in os.walk(root_dir):
    dirs[:] = [d for d in dirs if not d.startswith("MLP")]

    if "FINAL_metrics_test.csv" in files:
        full_path = os.path.join(root, "FINAL_metrics_test.csv")

        try:
            table = pd.read_csv(full_path)

            mae_row = table[table["Метрики"] == "MAE"].iloc[:, 1:]
            mape_row = table[table["Метрики"] == "R2"].iloc[:, 1:]

            res = {
                "Фичи": os.path.basename(root),
                "MAE": round(mae_row.mean(axis=1).values, 4),
                "R2": round(mape_row.mean(axis=1).values, 2),
            }
            average_results.append(res)
        except:
            continue

df_res = pd.DataFrame(average_results)

if not df_res.empty:
    df_res = df_res.sort_values(by="Фичи").reset_index(drop=True)

    df_res["Фичи"] = [
        "Скорости",
        "Скорости + поверхности",
        "Токи",
        "Токи + поверхности",
        "Токи + скорости + поверхности",
    ]

    df_res = df_res[["Фичи", "MAE", "R2"]]
    print(df_res)

    # Задаем глобальный увеличенный размер шрифта по умолчанию (для осей и названий признаков)
    plt.rcParams.update({"font.size": 14})

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7), sharey=True)

    df_mae = df_res.sort_values(by="MAE", ascending=True)
    bars_mae = axes.barh(
        df_mae["Фичи"], df_mae["MAE"], color="blue", edgecolor="black"
    )
    axes.set_title("MAE\n", fontsize=20, fontweight="bold")
    axes.set_xlabel("MAE", fontsize=20)
    axes.set_ylabel("Признаки", fontsize=15)
    axes.grid(axis="x", linestyle="--", alpha=0.7)
    axes.bar_label(bars_mae, fmt="%.4f", padding=5, fontsize=16)

    df_r2 = df_res.sort_values(by="R2", ascending=False)
    bars_r2 = axes.barh(
        df_r2["Фичи"], df_r2["R2"], color="red", edgecolor="black"
    )
    axes.set_title("R2\n", fontsize=20, fontweight="bold")
    axes.set_xlabel("R2", fontsize=20)
    axes.set_ylabel("")
    axes.grid(axis="x", linestyle="--", alpha=0.7)
    axes.bar_label(bars_r2, fmt="%.2f", padding=5, fontsize=16)

    # Увеличиваем размер шрифта для засечек (чисел) на осях X и Y
    axes.tick_params(axis="both", labelsize=20)
    axes.tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    chart_path = os.path.join(root_dir, "Features_Quality_Comparison.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

df_res.to_csv(
    os.path.join(root_dir, "Average_results_comparison.csv"),
    encoding="utf-8-sig",
    index=False,
)
