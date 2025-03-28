#%%
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.decomposition import PCA

#%%

class DataProcessor:
    """
    DataProcessor class to load, preprocess and standardise data
    
    Attributes:
    file_path (str): Path to the data file
    df (pd.DataFrame): Dataframe to store the data
    numeric_columns (list): List of numeric columns in the dataframe
    
    Methods:
    load_data(): Load the data from the file_path
    filter_numeric_columns(): Filter the dataframe to only include numeric columns
    rename_column(old_column_name, new_column_name): Rename a column in the dataframe
    Standardise(): Standardise the numeric columns in the dataframe
    """

    def __init__(self,file_path):
        # Initialise the path to the data file
        self.file_path=file_path
        self.df=None
        self.numeric_columns=None
    
    def load_data(self):
        # Load the data from the file_path
        self.df=pd.read_csv(self.file_path)

    def filter_numeric_columns(self):
        # Filter the dataframe to only include numeric columns
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        self.df=self.df[self.numeric_columns]
    
    def rename_column(self,old_column_name,new_column_name):
        # Rename a column in the dataframe
        self.df.rename(columns={old_column_name:new_column_name}, inplace=True)
    
    def Standardise(self):
        # Standardise the numeric columns in the dataframe
        scaler = StandardScaler()
        # Fit and transform the data for analysis
        data = scaler.fit_transform(self.df)
        # Convert the numpy array back to a dataframe for EDA
        df = pd.DataFrame(data, columns=self.df.columns)
        return df, data

class EDA:
    """
    EDA class to perform exploratory data analysis on the data
    
    Attributes:
    df (pd.DataFrame): Dataframe to store the data
    
    Methods:
    display_shape(): Display the shape of the data
    display_data_types(): Display the data types of the columns
    display_summary_statistics(): Display the summary statistics of the data
    display_missing_values(): Display the missing values in the data
    display_heatmap(): Display the heatmap of the correlation matrix
    display_distribution(): Display the distribution of the data
    display_boxplot(): Display the boxplot of the data
    display_pairplot(): Display the pairplot of the data
    """

    def __init__(self,df):
        # Initialise the dataframe
        self.df=df

    def display_shape(self):
        # Display data shape
        print('Data Shape:')
        print(self.df.shape)

    def display_data_types(self):
        # Display column data types
        print('Data Types:')
        print(self.df.dtypes)

    def display_summary_statistics(self):
        # Display summary statistics
        print('========== Summary Stats ==========')
        print(self.df.describe())

    def display_missing_values(self):
        # Display missing values
        print('Missing Values:')
        print(self.df.isnull().sum())
    
    def display_heatmap(self):
        # Display correlation heatmap
        plt.figure(figsize=(10,10))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
    
    def display_distribution(self):
        # Display data distribution
        self.df.hist(bins=20, figsize=(20,20))
        plt.suptitle('Distribution of Data')
        plt.show()
    
    def display_boxplot(self):
        # Display boxplot
        self.df.boxplot(figsize=(20,10))
        plt.title('Boxplot of Data')
        plt.show()
    
    def display_pairplot(self):
        # Display pairplot
        sns.pairplot(self.df)
        plt.suptitle('Pairplot of Data')
        plt.show()

from sklearn.decomposition import PCA

class RunPCA:
    def __init__(self, data, n_components=2):
        """
        RunPCA class to perform Principal Component Analysis on the data
        
        Attributes:
        data (pd.DataFrame): Dataframe containing the standardized data
        pca (PCA): PCA object to perform the analysis
        principal_components (np.array): Array to store the principal components
        """
        # Store the data passed as an argument
        self.data = data
        
        # Instantiate the scikit-learn PCA model
        self.pca = PCA(n_components=n_components)
        
        # Fit the model and transform the data
        self.principal_components = self.pca.fit_transform(self.data)

    def display_explained_variance(self):
        # Display the explained variance
        print("Explained Variance Ratio:")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_, start=1):
            print(f"PC{i}: {ratio:.2%}")
    
    def prepare_pca_results(self):
        # Prepare the PCA results as a DataFrame (optional)
        pc_df = pd.DataFrame(
            data=self.principal_components, 
            index=self.data.index, 
            columns=[f'PC{i+1}' for i in range(self.principal_components.shape[1])]
        )
        return pc_df
    
    def display_pca_scores(self, pc_df):
        # Display the PCA scores over the observations (if relevant)
        plt.figure(figsize=(10, 6))
        plt.scatter(pc_df['PC1'], pc_df['PC2'], c='grey', alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Scatter Plot of Numeric Features")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Data loading and preprocessing
    file_path = './datasets/IMDB-Movies.csv'
    data_processor = DataProcessor(file_path)
    data_processor.load_data()
    data_processor.filter_numeric_columns()
    data_processor.rename_column('Unnamed: 0', 'Ranking')
    df_std, data = data_processor.Standardise()
    
    # EDA
    eda = EDA(df_std)
    eda.display_shape()
    eda.display_data_types()
    eda.display_summary_statistics()
    eda.display_missing_values()
    eda.display_heatmap()
    eda.display_distribution()
    eda.display_boxplot()
    eda.display_pairplot()
    
    # PCA
    # Pass the standardised data to the PCA class
    pca_instance = RunPCA(data, n_components=2)
    pca_instance.display_explained_variance()
    pc_df = pca_instance.prepare_pca_results()
    pca_instance.display_pca_scores(pc_df)


# %%
