import os
import pandas as pd

root_dir = r"./Currents_to_Slippage/Seeking_for_best_features_05-05-2026_18-18-28"
average_results = []

for root, dirs, files in os.walk(root_dir):
    dirs[:] = [d for d in dirs if not d.startswith('MLP')]
    
    if "FINAL_metrics_test.csv" in files:
        full_path = os.path.join(root, "FINAL_metrics_test.csv")
        
        try:
            table = pd.read_csv(full_path)
            
            # Фильтруем строки по нужным метрикам
            mae_row = table[table['Метрики'] == 'MAE'].iloc[:, 1:]
            mape_row = table[table['Метрики'] == 'MAPE, %'].iloc[:, 1:]
            
            # Считаем среднее между всеми двигателями (по строке)
            res = {
                'Фичи': os.path.basename(root),
                'MAE': round(mae_row.mean(axis=1).values[0],4),
                'MAPE, %': round(mape_row.mean(axis=1).values[0], 2)
            }
            average_results.append(res)
        except:
            continue

df_res = pd.DataFrame(average_results)

if not df_res.empty:
    df_res = df_res[['Фичи', 'MAE', 'MAPE, %']]
    print(df_res)

df_res.to_csv(os.path.join(root_dir, "Average_results_comparison.csv"), encoding="utf-8-sig")