import pandas as pd

REQUIRED_COLUMNS = [
    "Ürün", "Kategori", "Talep", "BirimMaliyet", "DepoMaliyet",
    "StoktaVar", "MinStok", "MaxStok", "TeslimatSüresi",
    "RafOmru", "FireOranı", "SatışFiyatı"
]

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Eksik sütunlar: {missing}")
        return df
    except Exception as e:
        print(f"Hata: {e}")
        return None
