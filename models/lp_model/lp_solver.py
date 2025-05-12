import pandas as pd
import numpy as np
from scipy.optimize import linprog
from typing import Tuple
import pulp

def solve_lp(filepath: str) -> Tuple[pd.DataFrame, float]:
    df = pd.read_csv(filepath)
    
    # Veri doğrulama
    if df.isnull().any().any():
        raise ValueError("Veri setinde eksik değerler bulunuyor!")
    
    # Gerekli sütunların varlığını kontrol et
    required_columns = ["StoktaVar", "MaxStok", "MinStok", "Talep", "FireOranı", "BirimMaliyet", "DepoMaliyet"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik sütunlar: {missing_columns}")
    
    # Model oluştur
    model = pulp.LpProblem("Stok_Optimizasyonu", pulp.LpMinimize)
    
    # Değişkenler
    x = pulp.LpVariable.dicts("siparis", 
                            ((i) for i in range(len(df))),
                            lowBound=0)
    
    # Amaç fonksiyonu
    model += pulp.lpSum([x[i] * df.loc[i, "BirimMaliyet"] + 
                        (x[i] + df.loc[i, "StoktaVar"]) * df.loc[i, "DepoMaliyet"]
                        for i in range(len(df))])
    
    # Kısıtlar
    for i in range(len(df)):
        # Maksimum stok limiti
        model += x[i] + df.loc[i, "StoktaVar"] <= df.loc[i, "MaxStok"]
        
        # Minimum stok limiti
        model += x[i] + df.loc[i, "StoktaVar"] >= df.loc[i, "MinStok"]
        
        # Talep karşılama
        model += (x[i] + df.loc[i, "StoktaVar"]) * (1 - df.loc[i, "FireOranı"]) >= df.loc[i, "Talep"]
    
    # Modeli çöz
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Sonuçları işle
    df["SiparişMiktarı"] = [x[i].value() for i in range(len(df))]
    df["ToplamStok"] = df["SiparişMiktarı"] + df["StoktaVar"]
    df["ToplamMaliyet"] = df["SiparişMiktarı"] * df["BirimMaliyet"] + df["ToplamStok"] * df["DepoMaliyet"]
    
    # Tüm gerekli sütunları içeren sonuç DataFrame'i
    result_columns = ["Ürün", "Talep", "StoktaVar", "SiparişMiktarı", "ToplamStok", "ToplamMaliyet", "MaxStok", "MinStok"]
    return df[result_columns], model.objective.value()
