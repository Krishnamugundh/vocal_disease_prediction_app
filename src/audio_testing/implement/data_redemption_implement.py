#Data extraction
import os
import re
import wfdb
import numpy as np 
import pandas as pd
# from pathlib import Path
from audio_testing.DataTypes.data_entity import DataReductionInfo




class DataRedemption():
    def __init__(
        self,
        config_detail:DataReductionInfo
    ):
        # #creating new DIrectory
        '''
        l = os.path.basename(os.path.normpath(os.getcwd()))
        if(l == "Final_csv_data"):
            os.chdir("..")'''
            
        #defining Dataframe
        self.df1 = pd.DataFrame()
        
        # Change directory
        '''
        self.home = os.getcwd()
        self.csv_dir = os.path.join(os.getcwd(), "Final_csv_data")
        self.main_data_dir =  os.path.join(os.getcwd(), "voice-icar-federico-ii-database-1.0.0") '''

        # self.curr_home = 
        self.curr_data_dir = config_detail.data_path
        self.save_dir = config_detail.save_df_at
        self.save_name = config_detail.save_df_name
        self.format = config_detail.save_format

    # To identify the diagnoses the patient has undergone.
    def parse_comments(self,comments:str):  # sourcery skip: use-named-expression
        pattern = r'<age>: (\d+)\s+<sex>: (\w+)\s+<diagnoses>: ([^<]+)\s+<medications>: ([^<]+)'
        match = re.search(pattern, comments)
        if match:
            age, sex, diagnoses, medications = match.groups()
            return diagnoses.strip()
        else:
            return None
    # ------------------------------------------------------------------- #
    def retreive_data(self):
        # os.chdir(self.curr_home)
        # print("curr dir:",os.getcwd())
        try:
            # To get the .text files from dir and stire the audio amplitudes for each sample seperately.
            # List to store records information
            All_Id = []
            All_signal = []
            All_diagnose = []

            # os.chdir(self.curr_data_dir)
            # Loop through each .hea file in the directory
            print(f"Reading records from {self.curr_data_dir}")
            for filename in os.listdir(self.curr_data_dir):
                if filename.endswith('.hea'):
                    record_path = os.path.join(self.curr_data_dir, filename.replace('.hea',''))
                    
                    # Read the record
                    record = wfdb.rdrecord(record_path)
                    
                    #Getting the patient ID
                    name = record.record_name
                    All_Id.append(name)
                    
                    # diagnoses of Patient
                    diagnose = self.parse_comments(record.comments[0])
                    All_diagnose.append(diagnose)
                    
                    # Converts the signals into a single array for processing
                    signal_array = record.p_signal.reshape(-1)
                    All_signal.append(np.array(signal_array))
                    
            print("All data has been Successfully loaded into All_records")
            self.df1 = pd.DataFrame({
                "ID" : All_Id,
                "Signal" : All_signal,
                "diagnoses" : All_diagnose,})
            print("Data has been stored into a dataframe and returned to the variable assigned")
        except Exception as F:
            print(f"<--------------Something is wrong - msg:{F}--------------------->")
        return self.df1
        
        # --------------------------------------------------------------- #
    
    def store_dataframe(self):
        # Defining the dataframe path.
        self.save_name += f".{self.format}"
            
        saving_dir = os.path.join(self.save_dir + self.save_name)
        
        # If dir not exists, create it
        if not os.path.exists(self.save_dir):
            print(f"Creating directory:{os.getcwd()+self.save_dir}")
            os.makedirs(self.save_dir)
        
        # Save DataFrame
        self.df1.to_hdf(saving_dir,key='df_001', mode='w')

        print(f"saved the data from All_records as {self.format} : {self.save_name}")
        