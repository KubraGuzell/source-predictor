import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit

# Kendi yazdığın util fonksiyonları
from data_utils import load_dataset, basic_cleaning

DATA_PATH = "data/dataset.json"
OUTPUT_DIR = "data_splits"
RANDOM_STATE = 42

def create_clusters(df, text_col="text", similarity_threshold=0.7):
    """
    Metinleri TF-IDF ile vektörize eder ve DBSCAN kullanarak kümeler.
    similarity_threshold=0.7 -> %70 ve üzeri benzeyen metinler aynı ID'yi alır.
    """
    print("\n[1/3] TF-IDF vektörleri oluşturuluyor...")
    # max_features RAM'i korur, veri setindeki en önemli 15.000 kelimeyi alır
    vectorizer = TfidfVectorizer(max_features=15000)
    X = vectorizer.fit_transform(df[text_col])

    print(f"[2/3] DBSCAN ile benzer metinler kümeleniyor... (Bekleyin, biraz sürebilir)")
    # cosine distance = 1 - cosine similarity
    eps_distance = 1.0 - similarity_threshold 
    
    # min_samples=1 ÇOK KRİTİKTİR! 
    # Benzersiz her metnin kendi kümesini (ID'sini) oluşturmasını sağlar, 
    # böylece 'noise' (-1) kümesi oluşup alakasız metinleri aynı gruba toplamaz.
    clustering = DBSCAN(eps=eps_distance, min_samples=1, metric='cosine', n_jobs=-1)
    cluster_labels = clustering.fit_predict(X)
    
    return cluster_labels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Veri yükleniyor ve temizleniyor...")
    df = load_dataset(DATA_PATH)
    df = basic_cleaning(df)

    print(f"Toplam temizlenmiş satır: {len(df)}")
    
    # --- 1. ADIM: MATEMATİKSEL KÜMELEME ---
    df['cluster_id'] = create_clusters(df, text_col="text", similarity_threshold=0.70)
    
    unique_clusters = df['cluster_id'].nunique()
    print(f"\nMatematiksel Kümeleme Tamamlandı!")
    print(f"Tespit edilen benzersiz belge/küme sayısı: {unique_clusters}")
    print(f"Ortalama bir belge {len(df)/unique_clusters:.1f} cümleden/paragraftan oluşuyor.")

    # --- 2. ADIM: GRUP BAZLI BÖLME (Sızıntıyı Önleme) ---
    print("\n[3/3] Veri setleri GroupShuffleSplit ile ayrılıyor...")
    
    # DİKKAT: GroupShuffleSplit doğası gereği 'stratify' desteklemez. 
    # Veri sızıntısını önlemek için sınıf dengesinden küçük bir miktar feragat ediyoruz.
    
    # Önce Train (%70) ve Temp (%30) olarak ayır
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss_train.split(df, groups=df['cluster_id']))
    
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Sonra Temp verisini Val (%15) ve Test (%15) olarak ikiye böl
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df['cluster_id']))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Temizlik (cluster_id artık işimiz bitti, modele gitmesine gerek yok)
    train_df = train_df.drop(columns=['cluster_id'])
    val_df = val_df.drop(columns=['cluster_id'])
    test_df = test_df.drop(columns=['cluster_id'])

    print("-" * 30)
    print(f"Train boyutu: {len(train_df)}")
    print(f"Val boyutu  : {len(val_df)}")
    print(f"Test boyutu : {len(test_df)}")
    print("-" * 30)

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"Veri setleri sızıntısız şekilde kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()