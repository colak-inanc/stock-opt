import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from typing import Tuple, List
import multiprocessing

def evaluate(individual: List[float], df: pd.DataFrame) -> Tuple[float]:
    total_cost = 0
    penalty = 0
    
    for i, qty in enumerate(individual):
        stokta_var = float(df.loc[i, "StoktaVar"])
        toplam_stok = stokta_var + qty
        max_stok = float(df.loc[i, "MaxStok"])
        min_stok = float(df.loc[i, "MinStok"])
        talep = float(df.loc[i, "Talep"])
        fire_orani = float(df.loc[i, "FireOranı"])
        birim_maliyet = float(df.loc[i, "BirimMaliyet"])
        depo_maliyet = float(df.loc[i, "DepoMaliyet"])
        
        # Stok limiti kontrolü
        if toplam_stok > max_stok:
            penalty += (toplam_stok - max_stok) * birim_maliyet * 2
        
        # Minimum stok kontrolü
        if toplam_stok < min_stok:
            penalty += (min_stok - toplam_stok) * birim_maliyet * 2
        
        # Fire hesaplama
        fire = toplam_stok * fire_orani
        kalan_stok = toplam_stok - fire
        
        # Talep karşılama kontrolü
        if kalan_stok < talep:
            penalty += (talep - kalan_stok) * birim_maliyet * 3
        
        # Maliyet hesaplama
        malzeme_maliyeti = qty * birim_maliyet
        stok_maliyeti = toplam_stok * depo_maliyet
        total_cost += malzeme_maliyeti + stok_maliyeti
    
    return (total_cost + penalty,)

def create_toolbox(df: pd.DataFrame, pop_size: int) -> base.Toolbox:
    # Önceki creator tanımlamalarını temizle
    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin
    if 'Individual' in creator.__dict__:
        del creator.Individual
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Akıllı başlangıç popülasyonu
    def init_individual():
        individual = []
        for _, row in df.iterrows():
            stokta_var = float(row["StoktaVar"])
            talep = float(row["Talep"])
            max_stok = float(row["MaxStok"])
            min_stok = float(row["MinStok"])
            
            # Talep ve stok durumuna göre akıllı başlangıç değeri
            if stokta_var < talep:
                # Talep karşılanmıyorsa, talep kadar sipariş
                order = talep - stokta_var
            else:
                # Talep karşılanıyorsa, minimum stok seviyesine göre sipariş
                order = max(0, min_stok - stokta_var)
            
            # Rastgele varyasyon ekle
            variation = random.uniform(-0.1, 0.1) * order
            order = max(0, order + variation)
            
            # Maksimum stok limitini aşmama
            order = min(order, max_stok - stokta_var)
            
            individual.append(order)
        
        return individual
    
    # Birey oluşturma
    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Paralel değerlendirme
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # Genetik operatörler
    toolbox.register("evaluate", evaluate, df=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=7)
    
    return toolbox

def solve_ga(filepath: str, ngen: int = 300, pop_size: int = 200) -> Tuple[pd.DataFrame, float]:
    df = pd.read_csv(filepath)
    
    # Veri doğrulama
    if df.isnull().any().any():
        raise ValueError("Veri setinde eksik değerler bulunuyor!")
    
    # Gerekli sütunların varlığını kontrol et
    required_columns = ["StoktaVar", "MaxStok", "MinStok", "Talep", "FireOranı", "BirimMaliyet", "DepoMaliyet"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik sütunlar: {missing_columns}")
    
    toolbox = create_toolbox(df, pop_size)
    
    # Hall of Fame ve istatistikler
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    
    # Algoritma parametreleri
    cxpb, mutpb = 0.8, 0.1  # Daha yüksek crossover, daha düşük mutasyon
    
    # Başlangıç popülasyonu
    pop = toolbox.population(n=pop_size)
    
    try:
        # Algoritma çalıştırma
        pop, logbook = algorithms.eaSimple(
            pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
            stats=stats, halloffame=hof, verbose=True
        )
        
        # En iyi çözümü al
        best = hof[0]
        
        # Sonuçları işle
        df["SiparişMiktarı"] = np.round(best, 2)
        df["ToplamStok"] = df["SiparişMiktarı"] + df["StoktaVar"]
        df["ToplamMaliyet"] = df["SiparişMiktarı"] * df["BirimMaliyet"] + df["ToplamStok"] * df["DepoMaliyet"]
        
        # Sonuçları doğrula
        if (df["ToplamStok"] > df["MaxStok"]).any():
            print("⚠️ Uyarı: Bazı ürünler için maksimum stok limiti aşıldı!")
        
        if (df["ToplamStok"] < df["MinStok"]).any():
            print("⚠️ Uyarı: Bazı ürünler için minimum stok limiti altına düşüldü!")
        
        if (df["ToplamStok"] < df["Talep"]).any():
            print("⚠️ Uyarı: Bazı ürünler için talep karşılanamıyor!")
        
        # Tüm gerekli sütunları içeren sonuç DataFrame'i
        result_columns = ["Ürün", "Talep", "StoktaVar", "SiparişMiktarı", "ToplamStok", "ToplamMaliyet", "MaxStok", "MinStok"]
        return df[result_columns], evaluate(best, df)[0]
    
    except Exception as e:
        print(f"GA çözümü sırasında hata oluştu: {str(e)}")
        raise
    finally:
        # Multiprocessing havuzunu temizle
        if 'pool' in locals():
            pool.close()
            pool.join()
