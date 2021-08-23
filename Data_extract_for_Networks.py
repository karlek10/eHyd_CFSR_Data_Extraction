# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 07:43:04 2021

@author: karlo
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.dates as mdates

def extract_eHyd(file_path, list_files=None, num=0, start_date=None, end_date=None, avg=None):
    """
    Parameters
    ----------
    file_path : string - path to the file to procces
    list_files: list of all files in the folder
    num: number of the file in the list
    start_date: starting date
    end_date: ending date  

    Returns
    -------
    data : dataframe - variable value
        1st col dateindex, second col timeseries
    """
    df = pd.read_csv(file_path, encoding="windows-1250", sep=';', names = np.arange(0,3)) # read the file into pandas df 
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # strip whitespaces from the cols
    idx = df.index[df[0]=="Werte:"][0].astype("int") # find starting index of the data
    st_name = df[1][0] # find the name of the station
    data = df.iloc[idx+1:-1].drop(2, axis=1).replace(",", ".", regex=True).rename(columns={0:"Date", 1:st_name}) 
    while data.index[data[st_name] == "Lücke"].any(axis=0): # checks for missing data
        data.drop(data.index[data[st_name] == "Lücke"][0], axis = 0, inplace=True)
    data[st_name] = data[st_name].astype("float32") # converts values to float32
    data.index = pd.to_datetime(data["Date"], dayfirst=True) # converts dates to datetime format
    data = data.drop("Date", axis=1) # drops the date column
    data[st_name] = np.where(data[st_name] == 0.001, 0, data[st_name]) # changes missing values 0.001 with 0.0
    data.index = data.index.normalize()
    print ("Done converting {}/{} --> {}.".format(num+1, len(list_files),st_name))
    if start_date is not None and end_date is not None:
        data = data[(data.index >= start_date) & (data.index < end_date)] # start and stop date
        return data
    elif start_date is not None:
        data = data[(data.index >= start_date)] # start and stop date
        return data
    elif end_date is not None:
        data = data[(data.index < end_date)] # start and stop date
        return data
    else:
        return data

def extraction_main(function, basin, folder, start_date=None, end_date = None, avg=None): 
    """
    Parameters
    ----------
    function: function name
    basin: basin name
    folder : str - folder name
        Takes in the name of the folder (variable) to extract.
    Returns
    -------
    df_comp : dataframe - variable value
    """
    owd = os.getcwd() # gets the workign dir
    list_files = os.listdir("D:\\01_Raw_input_data\\" + basin + "/" + folder + "/") # creates a list of files in the dir
    os.chdir("D:\\01_Raw_input_data\\" + basin+ "/" + folder + "/") # changes working dir
    dict_frames= [function(file, list_files, num, start_date, end_date, avg) for num, file in enumerate(list_files)] # creates a dictionary of df-s
    os.chdir(owd) # returns to dir with the script
    df_comp = pd.concat(dict_frames, axis=1) # creates a df with all stations
    return df_comp
    
def extract_CFSR_temp(file_path, list_files=None, num=0, start_date=None, end_date=None, avg=None):
    """
    Parameters
    ----------
    file_path : string - path to the file to procces
    list_files: list of all files in the folder
    num: number of the file in the list
    start_date: starting date
    end_date: ending date  

    Returns
    -------
    data : dataframe - variable value
        1st col dateindex, second col timeseries
    """
    if any(char.isdigit() for char in file_path):
        st_name = file_path[1:-4]
        df = pd.read_csv(file_path, encoding="windows-1250", names=[st_name+"_t_MAX", st_name +"_t_MIN"])
        date = pd.to_datetime(str(int(df.loc[0][0])))
        dates = pd.date_range(date, periods=len(df)-1, freq="D")
        data = df.iloc[1:].set_index(dates)
        print ("Done converting {}/{} --> {}.".format(num+1, len(list_files),st_name))
        if start_date is not None and end_date is not None:
            data = data[(data.index >= start_date) & (data.index <= end_date)] # start and stop date
            return data
        elif start_date is not None:
            data = data[(data.index >= start_date)] # start and stop date
            return data
        elif end_date is not None:
            data = data[(data.index <= end_date)] # start and stop date
            return data
        else:
            return data
    
def extract_CFSR_else(file_path, list_files=None, num=0, start_date=None, end_date=None, avg=None):
    """
    Parameters
    ----------
    file_path : string - path to the file to procces
    list_files: list of all files in the folder
    num: number of the file in the list
    start_date: starting date
    end_date: ending date  

    Returns
    -------
    data : dataframe - variable value
        1st col dateindex, second col timeseries
    """
    if any(char.isdigit() for char in file_path):
        st_name = file_path[1:-4]
        meteo_measurement = file_path[0]
        df = pd.read_csv(file_path, encoding="windows-1250", names=[st_name+"_"+ meteo_measurement])
        date = pd.to_datetime(str(int(df.loc[0][0])))
        dates = pd.date_range(date, periods=len(df)-1, freq="D")
        data = df.iloc[1:].set_index(dates)
        print ("Done converting {}/{} --> {}.".format(num+1, len(list_files),st_name))
        if start_date is not None and end_date is not None:
            data = data[(data.index >= start_date) & (data.index <= end_date)] # start and stop date
            return data
        elif start_date is not None:
            data = data[(data.index >= start_date)] # start and stop date
            return data
        elif end_date is not None:
            data = data[(data.index <= end_date)] # start and stop date
            return data
        else:
            return data 
            
def extract_zamg_temp_solar(file_path, start_date=None, end_date=None, feat=None):
    """
    Parameters
    ----------
    file_path : string - path to the file

    Returns
    -------
    d : dictionary - dictionary of dataframes, each dataframe is one feature 
                     (variable)
    """
    df = pd.read_csv(file_path, sep=";")
    df["datum"] = pd.to_datetime(df["datum"])
    data = df.set_index("datum").pivot(columns="istnr").replace("---", "nan")
    if start_date is not None and end_date is not None:
        data = data[(data.index >= start_date) & (data.index < end_date)] # start and stop date
    elif start_date is not None:
        data = data[(data.index >= start_date)] # start and stop date
    elif end_date is not None:
        data = data[(data.index < end_date)] # start and stop date
    features = set([i[0] for i in data.columns.tolist()]) # set of columns (features)
    feature_type = [i.split(" ")[0] for i in features] # list of variable names
    d = {} 
    for feature, col in zip(feature_type, features):
        d[feature] = pd.DataFrame()        
        d[feature] = data[col].astype("float32").dropna(axis=1, thresh=len(data)-100) # drops column if more than 100 nan    
    d = {f:d[f] for f in feat}
    return d # returns the dictionary

def extract_ITA_p(file_path, list_files=None, num=0, start_date=None, end_date=None, avg=None):
    df = pd.read_excel(file_path, usecols=([1,2,3,4,5]), )
    st_name = df.iloc[5,1].split(" / ")[0]
    data = df.iloc[13:, :]
    data = data.rename(columns=lambda x: x.strip())
    data = data.rename({data.columns[1]:"Date", 
                        data.columns[2]:"Prec_"+st_name, 
                        data.columns[3]:"MIN_T_"+st_name,
                        data.columns[4]:"MAX_T_"+st_name}, axis=1)
    data = data[data.columns[1:]].set_index("Date")
    data.index = pd.to_datetime(data.index, dayfirst=True)
    data = data.replace(",", ".", regex=True)\
        .replace(" ---", np.NaN, regex=True).replace("---", np.NaN, regex=True)\
        .replace(np.NaN, 0, regex=True).astype("float32")    
    print ("Done converting {}/{} --> {}.".format(num+1, len(list_files),st_name))
    if start_date is not None and end_date is not None:
        data = data[(data.index >= start_date) & (data.index < end_date)] # start and stop date
        data_p = data.iloc[:, ::3]
        return data_p
    elif start_date is not None:
        data = data[(data.index >= start_date)] # start and stop date
        data_p = data.iloc[:, ::3]
        return data_p
    elif end_date is not None:
        data = data[(data.index < end_date)] # start and stop date
        data_p = data.iloc[:, ::3]
        return data_p
    else:
        data_p = data.iloc[:, ::3]
        return data_p
        
def extract_ITA_temp(file_path, list_files=None, num=0, start_date=None, end_date=None, avg=None):
    df = pd.read_excel(file_path, usecols=([1,2,3,4,5]), )
    st_name = df.iloc[5,1].split(" / ")[0]
    data = df.iloc[13:, :]
    data = data.rename(columns=lambda x: x.strip())
    data = data.rename({data.columns[1]:"Date", 
                        data.columns[2]:"Prec_"+st_name, 
                        data.columns[3]:"MIN_T_"+st_name,
                        data.columns[4]:"MAX_T_"+st_name}, axis=1)
    data = data[data.columns[1:]].set_index("Date")
    data.index = pd.to_datetime(data.index, dayfirst=True)
    data = data.replace(",", ".", regex=True)\
        .replace(" ---", np.NaN, regex=True).replace("---", np.NaN, regex=True)\
        .replace(np.NaN, 0, regex=True).astype("float32")    
    print ("Done converting {}/{} --> {}.".format(num+1, len(list_files),st_name))
    if start_date is not None and end_date is not None:
        data = data[(data.index >= start_date) & (data.index < end_date)] # start and stop date
        data_temp = data.iloc[:, 1::1]
        data_t = pd.DataFrame()
        if avg:
            data_t["AVG_T_"+st_name] = data_temp.mean(axis=1)
            return data_t
        return data_temp
    elif start_date is not None:
        data = data[(data.index >= start_date)] # start and stop date
        data_temp = data.iloc[:, 1::1]
        if avg:
            data_t["AVG_T_"+st_name] = data_temp.mean(axis=1)
            return data_t
        return data_temp
    elif end_date is not None:
        data = data[(data.index < end_date)] # start and stop date
        data_temp = data.iloc[:, 1::1]
        if avg:
            data_t["AVG_T_"+st_name] = data_temp.mean(axis=1)
            return data_t
        return data_temp
    else:
        data_temp = data.iloc[:, 1::1]
        if avg:
            data_t["AVG_T_"+st_name] = data_temp.mean(axis=1)
            return data_t
        return data_temp  
 

if __name__ == "__main__":
    
    """
    EXAMPLES of how to use functions:
    
    df = extract_eHyd_extract("D:\OneDrive\Python\\10_eHyd_CFSR_Data_Extraction\input_data\snow_depth\\SH-Tageswerte-103705.csv", list_files=["32",], start_date="2000-01-01")
    
    df = extract_eHyd("D:\\OneDrive\Python\\10_eHyd_CFSR_Data_Extraction\\input_data\\Sill\\rain\\N-Tagessummen-103085.csv", list_files=["32",], start_date="2000-01-01")
    
    temp = extraction_main(extract_CFSR_temp, basin = "Sill", folder = "temp_min_max_CFSR",)
    
    rel_hum = extraction_main(extract_CFSR_else, basin= "Sill", folder = "humidity_CFSR",)
     
    solar = extraction_main(extract_CFSR_else, basin="Sill", folder="solar_rad_CFSR")
       
       
    plot_timeseries(df) # change ylabel if not plotting runoff
     
    
    temp = extract_CFSR_temp("input_data/temp_min_max_CFSR/t470122.txt", )
    
    list containing datatypes for training, default are runoff and measured precipitation
    
    """
    
    
    # ====== PARAMETERS ======
    basin_name = "Isel"
    start_date = "2000-02-18" # including
    end_date = "2014-01-01" #excluding
    avg = True
    runoff_st = ["Lienz", "Hopfgarten i. Def.-Zwenewald", "Brühl"] # list of runoff station to include in output 
    # =======================
    
    
    
    hum_cfsr = extraction_main(extract_CFSR_else, basin= basin_name, folder = "humidity_CFSR", start_date=start_date, end_date=end_date)
    solar_cfsr = extraction_main(extract_CFSR_else, basin=basin_name, folder="solar_rad_CFSR", start_date=start_date, end_date=end_date)
    temp_cfsr = extraction_main(extract_CFSR_temp, basin = basin_name, folder = "temp_min_max_CFSR", start_date=start_date, end_date=end_date)
    wind_cfsr = extraction_main(extract_CFSR_else, basin=basin_name, folder="wind_CFSR", start_date=start_date, end_date=end_date)
    
    
    # ===== PREPARING METEOROLOGICAL DATA ======

    
    precip_ita = extraction_main(extract_ITA_p, basin=basin_name, folder = "rain_ITA", start_date=start_date, end_date=end_date)
    
    precip_aut = extraction_main(extract_eHyd, basin=basin_name, folder = "rain", start_date=start_date, end_date=end_date)
    #temp_ita = extraction_main(extract_ITA_temp, basin=basin_name, folder = "rain_ITA", start_date=start_date, end_date=end_date, avg=avg)
    
    snow_meas = extraction_main(extract_eHyd, basin=basin_name, folder = "snow_depth", start_date=start_date, end_date=end_date)
    
    # temp_zamg = extract_zamg_temp_solar("D:\OneDrive\Python\\10_eHyd_CFSR_Data_Extraction\input_data\ZAMG_data\\daten_20000101_20151231.csv", 
    #                                     start_date=start_date, end_date=end_date, feat=["t",])["t"][[17901, 19505, 17701]]
    # ==========================================
    
    # ===== PREPARING RUNOFF DATA ===============
    runoff_meas = extraction_main(extract_eHyd, basin=basin_name, folder = "runoff", start_date=start_date, end_date=end_date)
    # #change order of runoff gauges if outlet not first
    runoff_meas = runoff_meas[runoff_st]  
    # ==========================================
    
    # # CONCATENING TRAINIGN DATA
    df_name = "%s_data" % basin_name.lower()
    data = pd.concat([runoff_meas, precip_aut, precip_ita, temp_cfsr, hum_cfsr, solar_cfsr, wind_cfsr, snow_meas,], axis = 1)
    df_name = pd.DataFrame(data)

    # # enter file name for saving training data
    file_name = basin_name +"_meteo_data.csv"
    df_name.to_csv("D:\\02_Model_input_ConvLSTM\\METEO/"+ basin_name +"/" + file_name, encoding = "windows-1250")
    print ("Done. The data has been prepared for traning in file {}.".format(file_name))
        
    
    # b = eHyd_precip("D:\\OneDrive\\Python\\10_eHyd_Data_Extraction\input_data\\rain\\N-Tagessummen-102319.csv")    
    
