import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import os

class AdvancedEarthquakePredictionSystem:
    def __init__(self, shapefile_path, dataset_path, predictions_file="predictions.csv", performance_file="model_performance.csv"):
        self.earthquake_data = None
        self.processed_data = None
        self.turkey_map = gpd.read_file(shapefile_path)
        self.predictions_file = predictions_file
        self.performance_file = performance_file
        self.mlp_model = None
        self.mlp_performance_history = []
        self.best_mlp_model = None
        self.best_mlp_score = -np.inf
        self.dataset_path = dataset_path
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.original_enlem = None  # Orijinal enlem değerlerini saklamak için
        self.original_boylam = None  # Orijinal boylam değerlerini saklamak için

    def initialize_or_load_mlp_model(self, input_size):
        model_file = "D:\\YAPAYS\\mlp_model.pkl"
        best_model_file = "D:\\YAPAYS\\best_mlp_model.pkl"  # En iyi modelin dosya adı

        if os.path.exists(best_model_file):
            try:
                # En iyi modeli yükle
                with open(best_model_file, "rb") as f:
                    self.best_mlp_model = pickle.load(f)
                print("Önceden eğitilmiş en iyi MLP modeli yüklendi.")
                self.mlp_model = self.best_mlp_model # self.mlp_model'i en iyi model ile güncelle
                print("self.mlp_model, en iyi model ile güncellendi.")
            except Exception as e:
                print(f"Model yüklenirken hata oluştu: {e}")
                self.reset_mlp_model()
        else:
            self.reset_mlp_model()

    def reset_mlp_model(self):
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(50, 30, 10),
            max_iter=1,
            activation='relu',
            solver='adam',
            random_state=42,
            warm_start=True
        )
        self.best_mlp_model = self.mlp_model
        self.best_mlp_score = -np.inf
        print("Yeni MLP modeli başlatıldı ve en iyi model olarak ayarlandı.")

    def load_earthquake_data(self):
        try:
            self.earthquake_data = pd.read_csv(self.dataset_path)
            print(f"{len(self.earthquake_data)} deprem verisi yüklendi.")
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")

    def feature_engineering(self):
        if self.earthquake_data is None or len(self.earthquake_data) == 0:
            print("Veri bulunamadı.")
            return

        df = self.earthquake_data.copy()

        df['Olus tarihi'] = pd.to_datetime(df['Olus tarihi'], format='%Y.%m.%d')
        df['yıl'] = df['Olus tarihi'].dt.year
        df['ay'] = df['Olus tarihi'].dt.month

        df['enlem_kategorik'] = pd.cut(df['Enlem'], bins=10)
        df['boylam_kategorik'] = pd.cut(df['Boylam'], bins=10)

        df['derinlik_z_skoru'] = (df['Der(km)'] - df['Der(km)'].mean()) / df['Der(km)'].std()

        df['ortalama_komsu_buyukluk'] = df['ML'].rolling(window=5, min_periods=1).mean()

        df['enlem'] = df['Enlem'].copy()
        df['boylam'] = df['Boylam'].copy()
        df['derinlik'] = df['Der(km)'].copy()

        self.processed_data = df

    def prepare_model_data(self):
        if self.processed_data is None:
            print("Veri ön işlemesi yapılmamış.")
            return None, None, None, None

        features = [
            'enlem', 'boylam', 'derinlik',
            'derinlik_z_skoru', 'ortalama_komsu_buyukluk',
            'yıl', 'ay'
        ]

        X = self.processed_data[features].fillna(0)
        y = self.processed_data['ML'].fillna(self.processed_data['ML'].mean())

        # Orijinal enlem ve boylam değerlerini saklayın
        self.original_enlem = X['enlem'].copy()
        self.original_boylam = X['boylam'].copy()

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_ensemble_models(self):
        """Çoklu model eğitimi ve topluluk tahmini"""
        X_train, self.X_test, y_train, self.y_test = self.prepare_model_data()

        if X_train is None:
            print("Veri hazırlama başarısız.")
            return None, None, None

        # Ölçekleyiciyi burada tanımlayın ve eğitin
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(self.X_test)  # X_test'i de aynı ölçekleyici ile dönüştürün

        # Kaydedilmiş modelleri yüklemeyi dene
        try:
            with open("rf_model.pkl", "rb") as f:
                rf_model = pickle.load(f)
            print("Random Forest modeli yüklendi.")
            with open("gb_model.pkl", "rb") as f:
                gb_model = pickle.load(f)
            print("Gradient Boosting modeli yüklendi.")
        except FileNotFoundError:
            print("Kayıtlı modeller bulunamadı, modeller eğitiliyor...")
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Gradient Boosting
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

            # Kfold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Her model için çapraz doğrulama skorlarını hesapla
            rf_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
            gb_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

            print(f"Random Forest Çapraz Doğrulama Skoru: {-rf_scores.mean():.4f} (std: {rf_scores.std():.4f})")
            print(f"Gradient Boosting Çapraz Doğrulama Skoru: {-gb_scores.mean():.4f} (std: {gb_scores.std():.4f})")

            # Modelleri eğit
            rf_model.fit(X_train, y_train)
            gb_model.fit(X_train, y_train)

            # Eğitilmiş modelleri kaydet
            with open("rf_model.pkl", "wb") as f:
                pickle.dump(rf_model, f)
            print("Random Forest modeli kaydedildi.")
            with open("gb_model.pkl", "wb") as f:
                pickle.dump(gb_model, f)
            print("Gradient Boosting modeli kaydedildi.")

        # Tahminler
        rf_pred = rf_model.predict(self.X_test)
        gb_pred = gb_model.predict(self.X_test)

        # Sinir Ağı (MLPRegressor) eğitimi ve değerlendirmesi
        mlp_pred = self.train_and_evaluate_mlp(X_train, self.X_test, y_train, self.y_test)

        # Topluluk tahmini (ağırlıklı ortalama)
        ensemble_pred = (
                0.3 * rf_pred +
                0.3 * gb_pred +
                0.4 * mlp_pred
        )

        # y_pred'i ensemble_pred olarak atayalım:
        self.y_pred = ensemble_pred

        # Performans metrikleri
        ensemble_mae = mean_absolute_error(self.y_test, ensemble_pred)
        ensemble_mse = mean_squared_error(self.y_test, ensemble_pred)
        ensemble_r2 = r2_score(self.y_test, ensemble_pred)
        print("Topluluk Modeli Performansı:")
        print(f"MAE: {ensemble_mae}")
        print(f"MSE: {ensemble_mse}")
        print(f"R2 Skoru: {ensemble_r2}")

        return ensemble_pred, self.X_test, self.y_test

    def calculate_similarity(self, y_true, y_pred, X_test_df):
        # X_test_df artık fonksiyona argüman olarak geliyor
        y_pred_locations = X_test_df.copy()
        y_pred_locations = y_pred_locations.drop('predicted_magnitude', axis=1)

        # Enlem ve boylamları radyana çevir
        y_true_rad = np.radians(y_true[['enlem', 'boylam']].values)
        y_pred_rad = np.radians(y_pred_locations[['enlem', 'boylam']].values)  # y_pred_locations kullanılıyor ve .values ile numpy arrayine çevriliyor

        # Haversine mesafesini hesapla
        distances = haversine_distances(y_true_rad, y_pred_rad)
        avg_distance = np.mean(np.diag(distances))
        similarity_score = 1 / (1 + avg_distance)
        similarity_percentage = similarity_score * 100

        return similarity_percentage

    def visualize_predictions_side_by_side(self, predictions, X_test, y_test):
        if self.earthquake_data is None or predictions is None or X_test is None or y_test is None:
            print("Görselleştirme için yeterli veri bulunamadı.")
            return

        # X_test_df'i orijinal enlem ve boylam değerleri ile oluşturun:
        X_test_df = pd.DataFrame({
            'enlem': self.original_enlem.loc[y_test.index],
            'boylam': self.original_boylam.loc[y_test.index],
            'derinlik': X_test[:, 2],  # Ölçeklendirilmiş derinlik
            'derinlik_z_skoru': X_test[:, 3],  # Ölçeklendirilmiş derinlik_z_skoru
            'ortalama_komsu_buyukluk': X_test[:, 4],  # Ölçeklendirilmiş ortalama_komsu_buyukluk
            'yıl': X_test[:, 5],  # Ölçeklendirilmiş yıl
            'ay': X_test[:, 6]  # Ölçeklendirilmiş ay
        })

        # Haritalandırılacak maksimum deprem sayısını belirleyin
        max_deprem_sayisi = 1000  # İstediğiniz sayıyla değiştirin

        # Test verilerinden rastgele örnekler alın
        if len(X_test_df) > max_deprem_sayisi:
            random_indices = np.random.choice(X_test_df.index, size=max_deprem_sayisi, replace=False)
            X_test_df = X_test_df.loc[random_indices]
            y_test = y_test.loc[random_indices]
            predictions = predictions[random_indices]

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        self.turkey_map.plot(ax=axes[0], color='lightgrey', edgecolor='black')
        sizes_actual = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * 100 + 20
        scatter_actual = axes[0].scatter(
            X_test_df['boylam'],
            X_test_df['enlem'],
            s=sizes_actual,
            c=y_test,
            cmap='Reds',
            alpha=0.8,
            edgecolors='k'
        )
        axes[0].set_title('Gerçek Deprem Verileri (Test)')
        axes[0].set_xlabel('Boylam')
        axes[0].set_ylabel('Enlem')
        cbar_actual = fig.colorbar(scatter_actual, ax=axes[0], label='Deprem Büyüklüğü')

        self.turkey_map.plot(ax=axes[1], color='lightgrey', edgecolor='black')
        sizes_pred = (predictions - predictions.min()) / (predictions.max() - predictions.min()) * 100 + 20
        scatter_pred = axes[1].scatter(
            X_test_df['boylam'],
            X_test_df['enlem'],
            s=sizes_pred,
            c=predictions,
            cmap='Blues',
            alpha=0.8,
            edgecolors='k'
        )
        axes[1].set_title('Tahmin Edilen Deprem Verileri')
        axes[1].set_xlabel('Boylam')
        axes[1].set_ylabel('Enlem')
        cbar_pred = fig.colorbar(scatter_pred, ax=axes[1], label='Tahmin Edilen Büyüklük')

        # Benzerlik yüzdesini hesapla ve yazdır
        y_test_df = pd.DataFrame({'enlem': X_test_df['enlem'], 'boylam': X_test_df['boylam']})
        X_test_df['predicted_magnitude'] = predictions
        similarity_percentage = self.calculate_similarity(y_test_df, predictions, X_test_df)

   

        plt.tight_layout()
        plt.show()

    def run_prediction_pipeline(self):
        """Tüm tahmin sürecini çalıştır, iki haritayı yan yana görselleştir ve sonuçları kaydet"""
        self.load_earthquake_data()
        self.feature_engineering()
        X_train, self.X_test, y_train, self.y_test = self.prepare_model_data()
        self.initialize_or_load_mlp_model(X_train.shape[1]) # Modeli yükle
        # Modeli yüklemeden önce train_and_evaluate_mlp fonksiyonunu çağırmayın
        predictions, X_test, y_test = self.train_ensemble_models()

        if predictions is not None:
            print("Tahmin süreci başarıyla tamamlandı. Haritalar oluşturuluyor...")

            # İki haritayı yan yana görselleştir
            self.visualize_predictions_side_by_side(predictions, X_test, y_test)

            # Tahmin sonuçlarını kaydet
            self.save_predictions_to_csv(predictions, X_test, y_test)
            # Tahminleri tablo olarak göster
            self.display_predictions_table()

            # MLP performans geçmişini çiz
            self.plot_mlp_performance_history()
        else:
            print("Tahmin işlemi başarısız.")

    def save_predictions_to_csv(self, predictions, X_test, y_test, file_name="predictions.csv"):
        global predictions_df
        if predictions is None or X_test is None:
            print("Tahmin sonuçları kaydedilemedi. Veri eksik.")
            return

        # X_test_df'i orijinal enlem ve boylam değerleri ile oluşturun:
        X_test_df = pd.DataFrame({
            'enlem': self.original_enlem.loc[y_test.index],
            'boylam': self.original_boylam.loc[y_test.index],
            'derinlik': X_test[:, 2],  # Ölçeklendirilmiş derinlik
            'derinlik_z_skoru': X_test[:, 3],  # Ölçeklendirilmiş derinlik_z_skoru
            'ortalama_komsu_buyukluk': X_test[:, 4],  # Ölçeklendirilmiş ortalama_komsu_buyukluk
            'yıl': X_test[:, 5],  # Ölçeklendirilmiş yıl
            'ay': X_test[:, 6]  # Ölçeklendirilmiş ay
        })

        # Tahmin sonuçlarını ve ilgili özellikleri bir DataFrame'e dönüştür
        predictions_df = pd.DataFrame({
            'enlem': X_test_df['enlem'],
            'boylam': X_test_df['boylam'],
            'tahmin_buyukluk': predictions
        })

        predictions_df.to_csv(file_name, index=False)
        print(f"Tahmin sonuçları {file_name} dosyasına kaydedildi.")

    def visualize_magnitude_distribution_comparison_mlp(self, y_test, mlp_pred):
        plt.figure(figsize=(10, 6))
        plt.hist(y_test, bins=20, alpha=0.7, label='Gerçek Büyüklükler', color='blue')
        plt.hist(mlp_pred, bins=20, alpha=0.7, label='MLP Tahminleri', color='orange')
        plt.title('Gerçek ve MLP Tahminlerinin Büyüklük Dağılımları')
        plt.xlabel('Büyüklük')
        plt.ylabel('Frekans')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_magnitude_over_time_comparison_mlp(self, y_test, mlp_pred, X_test):
        # X_test_df'i orijinal enlem ve boylam değerleri ile oluşturun:
        X_test_df = pd.DataFrame({
            'enlem': self.original_enlem.loc[y_test.index],
            'boylam': self.original_boylam.loc[y_test.index],
            'derinlik': X_test[:, 2],  # Ölçeklendirilmiş derinlik
            'derinlik_z_skoru': X_test[:, 3],  # Ölçeklendirilmiş derinlik_z_skoru
            'ortalama_komsu_buyukluk': X_test[:, 4],  # Ölçeklendirilmiş ortalama_komsu_buyukluk
            'yıl': X_test[:, 5],  # Ölçeklendirilmiş yıl
            'ay': X_test[:, 6]  # Ölçeklendirilmiş ay
        })
        time_indexed_data = pd.DataFrame({'y_test': y_test, 'mlp_pred': mlp_pred, 'tarih': X_test_df.index})
        time_indexed_data = time_indexed_data.sort_index()

        plt.figure(figsize=(12, 6))
        plt.plot(time_indexed_data.index, time_indexed_data['y_test'], label='Gerçek Büyüklükler', color='blue',
                 alpha=0.7)
        plt.plot(time_indexed_data.index, time_indexed_data['mlp_pred'], label='MLP Tahminleri', color='orange',
                 alpha=0.7)
        plt.title('Gerçek ve MLP Tahminlerinin Zaman İçindeki Değişimi')
        plt.xlabel('Zaman')
        plt.ylabel('Büyüklük')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_error_distribution_mlp(self, y_test, mlp_pred):
        errors = y_test - mlp_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, color='orange', alpha=0.7)
        plt.title('MLP Modeli Tahmin Hatalarının Dağılımı')
        plt.xlabel('Hata (Gerçek - Tahmin)')
        plt.ylabel('Frekans')
        plt.grid(True)
        plt.show()

    def visualize_actual_vs_predicted_magnitudes_mlp(self, y_test, mlp_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, mlp_pred, alpha=0.5, c='orange')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='blue')
        plt.title('Gerçek vs. MLP Tahminleri')
        plt.xlabel('Gerçek Büyüklük')
        plt.ylabel('MLP Tahmini')
        plt.grid(True)
        plt.show()

    def train_and_evaluate_mlp(self, X_train, X_test, y_train, y_test, max_iterations=1000):
        if self.mlp_model is None:
            self.initialize_or_load_mlp_model(X_train.shape[1])

        param_grid = {
            'hidden_layer_sizes': [(50, 50, 50), (100, 50, 20), (50, 30, 10)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [1000   , 1000, 2000]
        }



        grid_search = GridSearchCV(estimator=self.mlp_model, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        self.best_mlp_model = grid_search.best_estimator_
        self.best_mlp_score = -grid_search.best_score_

        with open("best_mlp_model.pkl", "wb") as f:
            pickle.dump(self.best_mlp_model, f)
        print("En iyi MLP modeli 'best_mlp_model.pkl' dosyasına kaydedildi.")

        for i in range(max_iterations):
            self.mlp_model.partial_fit(X_train, y_train)

            mlp_pred = self.mlp_model.predict(X_test)
            r2 = r2_score(y_test, mlp_pred)

            mae = mean_absolute_error(y_test, mlp_pred)
            mse = mean_squared_error(y_test, mlp_pred)
            self.mlp_performance_history.append({'MAE': mae, 'MSE': mse, 'R2': r2})

            if r2 > self.best_mlp_score:
                self.best_mlp_score = r2
                self.best_mlp_model = self.mlp_model
                print(f"İterasyon {i + 1}: Yeni en iyi R2 skoru: {r2:.4f}")

        mlp_pred = self.best_mlp_model.predict(X_test)
        mae = mean_absolute_error(y_test, mlp_pred)
        mse = mean_squared_error(y_test, mlp_pred)
        r2 = r2_score(y_test, mlp_pred)

        print("En İyi MLP Modeli Performansı:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Skoru: {r2:.4f}")

        self.visualize_magnitude_distribution_comparison_mlp(y_test, mlp_pred)
        self.visualize_magnitude_over_time_comparison_mlp(y_test, mlp_pred, X_test)
        self.visualize_error_distribution_mlp(y_test, mlp_pred)
        self.visualize_actual_vs_predicted_magnitudes_mlp(y_test, mlp_pred)

        pd.to_pickle(self.mlp_model, "mlp_model.pkl")
        pd.to_pickle(self.best_mlp_model, "best_mlp_model.pkl")

        return mlp_pred

    def plot_mlp_performance_history(self):
        """MLP modelinin performans geçmişini çizer."""
        if not self.mlp_performance_history:
            print("MLP performans geçmişi bulunamadı.")
            return

        history_df = pd.DataFrame(self.mlp_performance_history)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.plot(history_df['MAE'])
        plt.title('MAE (Ortalama Mutlak Hata)')
        plt.xlabel('İterasyon')
        plt.ylabel('MAE')

        plt.subplot(1, 3, 2)
        plt.plot(history_df['MSE'])
        plt.title('MSE (Ortalama Karesel Hata)')
        plt.xlabel('İterasyon')
        plt.ylabel('MSE')

        plt.subplot(1, 3, 3)
        plt.plot(history_df['R2'])
        plt.title('R2 Skoru')
        plt.xlabel('İterasyon')
        plt.ylabel('R2')

        plt.tight_layout()
        plt.show()



    
    def display_predictions_table(self):
        global predictions_df
        """Tahminleri tablo olarak gösterir."""
        if self.y_pred is None or self.X_test is None:
            print("Tahminler veya test verileri bulunamadı.")
            return
         # Tabloyu string'e çevir ve dosyaya kaydet
        table_string = predictions_df.to_string()
        with open("predictions_table.txt", "w") as f:
            f.write(table_string)
        print("Tahmin tablosu 'predictions_table.txt' dosyasına kaydedildi.")

        # X_test_df'i orijinal enlem ve boylam değerleri ile oluşturun:
        X_test_df = pd.DataFrame({
            'enlem': self.original_enlem.loc[self.y_test.index],
            'boylam': self.original_boylam.loc[self.y_test.index],
            'derinlik': self.X_test[:, 2],  # Ölçeklendirilmiş derinlik
            'derinlik_z_skoru': self.X_test[:, 3],  # Ölçeklendirilmiş derinlik_z_skoru
            'ortalama_komsu_buyukluk': self.X_test[:, 4],  # Ölçeklendirilmiş ortalama_komsu_buyukluk
            'yıl': self.X_test[:, 5],  # Ölçeklendirilmiş yıl
            'ay': self.X_test[:, 6]  # Ölçeklendirilmiş ay
        })

        # Tahminleri ve gerçek değerleri birleştir
        predictions_df = pd.DataFrame({
            'Enlem': X_test_df['enlem'],
            'Boylam': X_test_df['boylam'],
            'Gerçek Büyüklük': self.y_test,
            'Tahmin Edilen Büyüklük': self.y_pred
        })

        # Pandas'ın gösterim ayarlarını değiştirerek tüm satırları göster
        with pd.option_context('display.max_rows', None):
            print(predictions_df)

        # Tabloyu göster
        print(predictions_df)

if __name__ == "__main__":
    shapefile_path = "D:\YAPAYS\Yapay Sinir Ağları Projesi\gadm41_TUR_shp\gadm41_TUR_0.shp"
    dataset_path = "D:\\YAPAYS\\proje2\\veriler1.csv"
    system = AdvancedEarthquakePredictionSystem(shapefile_path, dataset_path)
    system.reset_mlp_model()
    system.run_prediction_pipeline()