import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_metrics = {"Дельты + скорости" : (0.07855346, 0.7943928),
             "Дельты + скорости + токи": (0.05340431,0.9219129),
             "Дельты + скорости + проскальзывания" : (0.07088134,0.84016734),
             "Дельты + скорости + проскальзывания\n + токи" : (0.056262817,0.91488034),
             "Дельты + скорости + проскальзывания\n + токи + поверхности" : (0.049674183, 0.9452299),
             "Дельты + скорости + проскальзывания\n + токи + поверхности + координаты" : (0.0036239862,0.99954194)
             }

Y_metrics = {"Дельты + скорости" : (0.21643025,0.19708723),
             "Дельты + скорости + токи": (0.21874657,0.1852507),
             "Дельты + скорости + проскальзывания" : (0.22260599,0.13126826),
             "Дельты + скорости + проскальзывания\n + токи" : (0.22353533,0.13841623),
             "Дельты + скорости + проскальзывания\n + токи + поверхности" : (0.20964664, 0.26519537),
             "Дельты + скорости + проскальзывания\n + токи + поверхности + координаты" : (0.003953374,0.9951181)
             }

Phi_metrics = {"Дельты + скорости" : (16.693647, 0.7781867),
             "Дельты + скорости + токи": (20.182205,0.7315006),
             "Дельты + скорости + проскальзывания" : (19.78859,0.7214731),
             "Дельты + скорости + проскальзывания\n + токи" : (23.491083,0.6857666),
             "Дельты + скорости + проскальзывания\n + токи + поверхности" : (21.023605,0.6856195),
             "Дельты + скорости + проскальзывания\n + токи + поверхности + координаты" : (0.0036239862*180/np.pi,0.99954194)
                }
print(Phi_metrics["Дельты + скорости + проскальзывания\n + токи + поверхности + координаты"])
home_direct = r"RNN"
os.makedirs(home_direct, exist_ok=True)

def save_separate_metrics(metrics_dict, title_name):
    labels = list(metrics_dict.keys())
    mae_vals = [val[0] for val in metrics_dict.values()]
    r2_vals = [val[1] for val in metrics_dict.values()]
    y_pos = np.arange(len(labels))
    
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars1 = ax1.barh(y_pos, mae_vals, color='crimson', edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel('Значение MAE')
    ax1.set_title(f'{title_name}: MAE', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    ax1.bar_label(bars1, fmt='%.4f', padding=5)
    plt.tight_layout()
    fig1.savefig(os.path.join(home_direct, f'{title_name}_MAE.png'), dpi=300)
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars2 = ax2.barh(y_pos, r2_vals, color='royalblue', edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Значение R2')
    ax2.set_title(f'{title_name}: R2', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    ax2.bar_label(bars2, fmt='%.4f', padding=5)
    plt.tight_layout()
    fig2.savefig(os.path.join(home_direct, f'{title_name}_R2.png'), dpi=300)
    plt.close(fig2)

save_separate_metrics(X_metrics, "X")
save_separate_metrics(Y_metrics, "Y")
save_separate_metrics(Phi_metrics, "Phi")
