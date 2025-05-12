import os
import pandas as pd
import numpy as np
from datetime import datetime
from models.lp_model.lp_solver import solve_lp
from models.ga_model.ga_solver import solve_ga
import json
import matplotlib.pyplot as plt

def get_next_experiment_dir(save_dir="reports"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    existing = [d for d in os.listdir(save_dir) if d.startswith("experiment_")]
    nums = []
    for d in existing:
        try:
            num = int(d.replace("experiment_", ""))
            nums.append(num)
        except ValueError:
            continue
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(save_dir, f"experiment_{next_num}")

def create_visualizations(lp_result, ga_result, experiment_dir):
    # Stil ayarları
    plt.style.use('default')
    
    # Veri doğrulama
    required_columns = ["Ürün", "Talep", "StoktaVar", "SiparişMiktarı", "ToplamStok", "ToplamMaliyet", "MaxStok", "MinStok"]
    for col in required_columns:
        if col not in lp_result.columns or col not in ga_result.columns:
            raise ValueError(f"Görselleştirme için gerekli sütun eksik: {col}")
    
    # 1. Sipariş Miktarları Karşılaştırması
    plt.figure(figsize=(15, 6))
    x = np.arange(len(lp_result))
    width = 0.35
    
    plt.bar(x - width/2, lp_result['SiparişMiktarı'], width, label='LP Modeli', color='#2ecc71')
    plt.bar(x + width/2, ga_result['SiparişMiktarı'], width, label='GA Modeli', color='#3498db')
    
    plt.xlabel('Ürünler')
    plt.ylabel('Sipariş Miktarı')
    plt.title('Ürünlere Göre Sipariş Miktarları Karşılaştırması')
    plt.xticks(x, lp_result['Ürün'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/siparis_karsilastirma.png')
    plt.close()
    
    # 2. Maliyet Dağılımı
    plt.figure(figsize=(10, 6))
    costs = pd.DataFrame({
        'Model': ['LP', 'GA'],
        'Maliyet': [lp_result['ToplamMaliyet'].sum(), ga_result['ToplamMaliyet'].sum()]
    })
    
    bars = plt.bar(costs['Model'], costs['Maliyet'], color=['#2ecc71', '#3498db'])
    plt.title('Model Bazında Toplam Maliyet Karşılaştırması')
    
    # Değerleri barların üzerine yaz
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f} TL',
                ha='center', va='bottom')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/maliyet_karsilastirma.png')
    plt.close()
    
    # 3. Stok Durumu Analizi
    plt.figure(figsize=(15, 6))
    
    # Stok oranlarını hesapla
    lp_stock_ratio = lp_result['ToplamStok'] / lp_result['MaxStok']
    ga_stock_ratio = ga_result['ToplamStok'] / ga_result['MaxStok']
    
    # Sıfıra bölme kontrolü
    if (lp_result['MaxStok'] == 0).any() or (ga_result['MaxStok'] == 0).any():
        print("⚠️ Uyarı: Bazı ürünler için maksimum stok değeri 0!")
        lp_stock_ratio = lp_stock_ratio.replace([np.inf, -np.inf], np.nan)
        ga_stock_ratio = ga_stock_ratio.replace([np.inf, -np.inf], np.nan)
    
    plt.plot(x, lp_stock_ratio, 'g-', label='LP Stok Oranı', linewidth=2)
    plt.plot(x, ga_stock_ratio, 'b-', label='GA Stok Oranı', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', label='Maksimum Stok Limiti')
    
    plt.xlabel('Ürünler')
    plt.ylabel('Stok/Maksimum Stok Oranı')
    plt.title('Ürünlere Göre Stok Doluluk Oranı')
    plt.xticks(x, lp_result['Ürün'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{experiment_dir}/stok_durumu.png')
    plt.close()

def run_experiments(data_path="data/demand_data.csv", save_dir="reports"):
    # Klasör oluştur
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #experiment_dir = os.path.join(save_dir, f"experiment_{timestamp}")
    experiment_dir = get_next_experiment_dir(save_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Veri setini yükle ve doğrula
    df = pd.read_csv(data_path)
    print(f"\n📊 Veri Seti Özeti:")
    print(f"Toplam Ürün Sayısı: {len(df):,}")
    print(f"Toplam Talep: {df['Talep'].sum():,.2f}")
    print(f"Toplam Stok: {df['StoktaVar'].sum():,.2f}")
    print(f"Ortalama Birim Maliyet: {df['BirimMaliyet'].mean():,.2f}")
    
    # LP çözümü
    print("\n▶ Linear Programming çözümü başlatılıyor...")
    start_time = datetime.now()
    lp_result, lp_cost = solve_lp(data_path)
    lp_time = (datetime.now() - start_time).total_seconds()
    
    # GA çözümü
    print("\n▶ Genetic Algorithm çözümü başlatılıyor...")
    start_time = datetime.now()
    ga_result, ga_cost = solve_ga(data_path)
    ga_time = (datetime.now() - start_time).total_seconds()
    
    # Sonuçları kaydet
    lp_result.to_csv(f"{experiment_dir}/lp_results.csv", index=False)
    ga_result.to_csv(f"{experiment_dir}/ga_results.csv", index=False)
    
    # Detaylı karşılaştırma
    comparison = pd.DataFrame({
        "Model": ["Linear Programming", "Genetic Algorithm"],
        "Toplam Maliyet (TL)": [lp_cost, ga_cost],
        "Çözüm Süresi (sn)": [lp_time, ga_time],
        "Ortalama Sipariş Miktarı": [
            lp_result["SiparişMiktarı"].mean(),
            ga_result["SiparişMiktarı"].mean()
        ],
        "Maksimum Sipariş Miktarı": [
            lp_result["SiparişMiktarı"].max(),
            ga_result["SiparişMiktarı"].max()
        ],
        "Minimum Sipariş Miktarı": [
            lp_result["SiparişMiktarı"].min(),
            ga_result["SiparişMiktarı"].min()
        ]
    })
    
    comparison.to_csv(f"{experiment_dir}/model_comparison.csv", index=False)
    
    # Performans metrikleri
    metrics = {
        "Maliyet Farkı (%)": abs(lp_cost - ga_cost) / min(lp_cost, ga_cost) * 100 if min(lp_cost, ga_cost) != float('inf') else float('inf'),
        "Hızlı Model": "LP" if lp_time < ga_time else "GA",
        "Hız Farkı (sn)": abs(lp_time - ga_time),
        "En İyi Model": "LP" if lp_cost < ga_cost else "GA",
        "Toplam Çözüm Süresi (sn)": lp_time + ga_time
    }
    
    with open(f"{experiment_dir}/performance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Görselleştirmeleri oluştur
    create_visualizations(lp_result, ga_result, experiment_dir)
    
    # Sonuçları yazdır
    print("\n📝 Karşılaştırma Sonuçları:")
    print("\nModel Performansı:")
    print(comparison.to_string(index=False))
    
    print("\nPerformans Metrikleri:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:,.2f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n✨ Sonuçlar kaydedildi: {experiment_dir}")
    print(f"📊 Görselleştirmeler oluşturuldu: {experiment_dir}/*.png")

if __name__ == "__main__":
    run_experiments()
