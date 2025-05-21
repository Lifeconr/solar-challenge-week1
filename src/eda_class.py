import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SolarEDA:
    """
    A class to perform Exploratory Data Analysis (EDA) on solar datasets.
    """

    def __init__(self, data_path, key_cols=None):
        """
        Initialize the SolarEDA class with a dataset.
        """
        self.df = pd.read_csv(data_path)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.key_cols = key_cols if key_cols else ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']

    def summary_statistics(self, numeric_cols):
        print("Summary Statistics:")
        summary = self.df[numeric_cols].describe()
        print(summary)
        return summary

    def missing_values_report(self):
        print("\nMissing Values:")
        missing = self.df.isna().sum()
        print(missing)
        print("\nColumns with >5% Missing:")
        missing_percent = (missing / len(self.df)) * 100
        print(missing_percent[missing_percent > 5])
        return missing

    def detect_outliers(self):
        outliers = {}
        for col in self.key_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            count = sum(z_scores > 3)
            print(f"\nOutliers in {col} (|Z| > 3): {count}")
            outliers[col] = count
        return outliers

    def clean_data(self, output_path):
        for col in self.key_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        df_clean = self.df.dropna(subset=['GHI', 'DNI', 'DHI'])
        print(f"\nCleaned Data Shape: {df_clean.shape}")
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        return df_clean

    def time_series_analysis(self):
        plt.figure(figsize=(10, 5))
        self.df.plot(x='Timestamp', y=['GHI', 'DNI', 'DHI', 'Tamb'], ax=plt.gca(), title='GHI, DNI, DHI, Tamb Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.show()
        self.df['Month'] = self.df['Timestamp'].dt.month
        monthly_avg = self.df.groupby('Month')[['GHI', 'DNI', 'DHI']].mean()
        monthly_avg.plot(kind='bar', figsize=(8, 5), title='Monthly Average GHI, DNI, DHI')
        plt.xlabel('Month')
        plt.ylabel('Irradiance (W/m²)')
        plt.show()

    def cleaning_impact(self):
        cleaning_impact = self.df.groupby('Cleaning')[['ModA', 'ModB']].mean()
        cleaning_impact.plot(kind='bar', figsize=(8, 5), title='ModA/ModB Pre/Post Cleaning')
        plt.xlabel('Cleaning (0 = No, 1 = Yes)')
        plt.ylabel('Irradiance (W/m²)')
        plt.show()

    def correlation_analysis(self, corr_cols):
        print("\nCorrelation Matrix:")
        corr_matrix = self.df[corr_cols].corr()
        print(corr_matrix)
        return corr_matrix

    def scatter_plots(self):
        self.df.plot.scatter(x='WS', y='GHI', figsize=(6, 4), title='WS vs. GHI')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('GHI (W/m²)')
        plt.show()
        self.df.plot.scatter(x='RH', y='Tamb', figsize=(6, 4), title='RH vs. Tamb')
        plt.xlabel('Relative Humidity (%)')
        plt.ylabel('Tamb (°C)')
        plt.show()
        self.df.plot.scatter(x='RH', y='GHI', figsize=(6, 4), title='RH vs. GHI')
        plt.xlabel('Relative Humidity (%)')
        plt.ylabel('GHI (W/m²)')
        plt.show()

    def distribution_analysis(self):
        self.df['GHI'].hist(bins=20, figsize=(6, 4), edgecolor='black')
        plt.title('GHI Distribution')
        plt.xlabel('GHI (W/m²)')
        plt.ylabel('Frequency')
        plt.show()
        self.df['WS'].hist(bins=20, figsize=(6, 4), edgecolor='black')
        plt.title('Wind Speed Distribution')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Frequency')
        plt.show()

    def temperature_humidity_analysis(self):
        """
        Analyze the impact of relative humidity on GHI.
        
        """
        print("\nAverage GHI by RH Range:")
        self.df['RH_bin'] = pd.cut(self.df['RH'], bins=[0, 25, 50, 75, 100], labels=['0-25', '25-50', '50-75', '75-100'])
        avg_ghi_by_rh = self.df.groupby('RH_bin')['GHI'].mean()
        print(avg_ghi_by_rh)
        return avg_ghi_by_rh
