import pandas as pd
import numpy as np
import os

class CMAPSSDataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        self.columns = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        
    def load_data(self, dataset_id='FD001'):
        """
        Loads train, test and RUL data for a specific dataset ID.
        """
        train_file = os.path.join(self.data_path, f'train_{dataset_id}.txt')
        test_file = os.path.join(self.data_path, f'test_{dataset_id}.txt')
        rul_file = os.path.join(self.data_path, f'RUL_{dataset_id}.txt')
        
        train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=self.columns)
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=self.columns)
        rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
        
        return train_df, test_df, rul_df
    
    def calculate_rul(self, train_df):
        """
        Calculates Remaining Useful Life (RUL) for training data.
        RUL = Max Cycle - Current Cycle
        """
        # Get max cycle for each unit
        max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycle']
        
        # Merge max cycle back to train_df
        train_df = train_df.merge(max_cycles, on='unit_number', how='left')
        
        # Calculate RUL
        train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
        
        # Drop max_cycle as it's no longer needed
        train_df.drop('max_cycle', axis=1, inplace=True)
        
        return train_df
    
    def preprocess_test_rul(self, test_df, rul_truth_df):
        """
        Calculates RUL for the test set. 
        Note: The RUL file contains the RUL at the LAST recorded cycle for each engine in the test set.
        So we need to add the current cycle to get the total life, etc.
        Actually, for test set, we often just want to predict the RUL at the END of the sequence.
        But to have a timeline, we can calculate back.
        
        For simplicity in this initial version, we will focus on predicting RUL for the *last* record of each engine in test,
        or we can construct full RUL target if needed for validation.
        
        Let's construct the True RUL for the test set for evaluation purposes.
        Test set RUL provided is for the last data point.
        """
        # Get max cycle in test data for each unit
        max_test_cycles = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_test_cycles.columns = ['unit_number', 'max_cycle']
        
        # The provided RUL is the remaining life AFTER the last cycle.
        rul_truth_df['unit_number'] = rul_truth_df.index + 1
        
        # We can calculate the RUL for every row in test_df
        # True RUL at time t = (True RUL at last cycle) + (Last cycle - t)
        
        test_df = test_df.merge(max_test_cycles, on='unit_number', how='left')
        test_df = test_df.merge(rul_truth_df, on='unit_number', how='left')
        
        test_df['RUL'] = test_df['RUL'] + (test_df['max_cycle'] - test_df['time_in_cycles'])
        
        test_df.drop('max_cycle', axis=1, inplace=True)
        
        return test_df

    def process_data(self, dataset_id='FD001'):
        train_df, test_df, rul_df = self.load_data(dataset_id)
        
        train_df = self.calculate_rul(train_df)
        test_df = self.preprocess_test_rul(test_df, rul_df)
        
        return train_df, test_df
