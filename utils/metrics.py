import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models(lp_path, ga_path, save_path="reports/comparison_plot.png"):
    lp = pd.read_csv(lp_path)
    ga = pd.read_csv(ga_path)

    df = pd.DataFrame({
        "Ürün": lp["Ürün"],
        "LP_Maliyet": lp["ToplamMaliyet"],
        "GA_Maliyet": ga["ToplamMaliyet"]
    })

    df_melted = df.melt(id_vars="Ürün", var_name="Model", value_name="Maliyet")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Ürün", y="Maliyet", hue="Model")
    plt.title("Model Bazında Ürün Maliyet Karşılaştırması")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✔ Grafik kaydedildi: {save_path}")
