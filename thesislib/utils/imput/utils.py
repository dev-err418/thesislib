import os
import pandas as pd
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse as date_parser
from dateutil.relativedelta import relativedelta
import math


# utils
def get_output_folder():
    """
    Returns the output folder where generated data should be stored
    """
    output_folder = os.path.join(os.getcwd(), "results")
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    return output_folder

def _count_key(label_count):
    return label_count[2]

def _print_list(obj):
    """
    Pretty list printing
    """
    for item in obj:
        print(item)

def form_design_matrix(observation_db):
    """
    Using the observations, builds a 'design matrix' that would hold all the observed
    features!
    """
    if type(observation_db) == "str":
        with open(observation_db) as fp:
            res = json.load(fp)
        observation_db = res
    data_matrix = {
        "condition_code": [],
        "condition_start": [],
        "condition_stop": [],
        "patient_id": [],
        "encounter_id": [],
        "patient_age": [],
        "marital_status": [],
        "race": [],
        "ethnicity": [],
        "gender": [],
    }

    data_matrix.update({k: [] for k in observations_db.keys()})
    return data_matrix

def get_design_matrix(observation_db, combined_df):
    """
    using a dataframe that combines the selected conditions, encounters and patients,
    this function fills up the design matrix with the proper values!
    """
    data_matrix = form_design_matrix(observation_db)
    
    grouped = reduced_combined.groupby(["ENCOUNTER", "CODE"])
    
    data_keys = list(data_matrix.keys())
    for item, df in grouped.__iter__():
        vector = {k: np.nan for k in data_keys}
        vector["encounter_id"] = item[0]
        vector["condition_code"] = item[1]
        vector["condition_start"] = df["START"].iloc[0]
        vector["condition_stop"] = df["STOP"].iloc[0]
        vector["patient_id"] = df["PATIENT"].iloc[0]
        vector["marital_status"] = df["MARITAL"].iloc[0]
        vector["race"] = df["RACE"].iloc[0]
        vector["ethnicity"] = df["ETHNICITY"].iloc[0]
        vector["gender"] = df["GENDER"].iloc[0]

        # fill in the observations
        for idx, obv_code in df["CODE_obv"].items():
            if obv_code not in data_keys:
                continue
            vector[obv_code] = df["VALUE"].loc[idx]

        # handle the age
        start_encounter_date = date_parser(df["START_enc"].iloc[0])
        patient_birthdate = date_parser(df["BIRTHDATE"].iloc[0])
        vector["patient_age"] = abs(patient_birthdate.year - start_encounter_date.year)

        for k,v in vector.items():
            data_matrix[k].append(v)
    return data_matrix

def _condition_transform_fxn(value, labels={}):
    return labels[value]

def _gender_transform_fxn(value):
    if value == 'M':
        return 1 # encoding for Male
    elif value == 'F':
        return 0 # encoding for female
    else:
        return 2 # encode the nan's

def _marital_transform_fxn(value):
    if value == 'M':
        return 1 # encoding for Married
    elif value == 'S':
        return 0 # encoding for single
    else:
        return 2 # encode the nan's

def _race_transform_fxn(value):
    if value == 'white':
        return 0
    elif value == 'black':
        return 1
    elif value == 'asian':
        return 2
    elif value == 'native':
        return 3
    elif value == 'other':
        return 4
    else:
        return value # nan's ?? there didn;t seem to be any though
    
def get_na_marital():
    """
    returns an encoding value for NaN values for marital status
    """
    return 2

def get_nan_value(obv_code):
    if obv_code == "2339-0":
        # glucose
        return -100 # typical valid values are positive and >50
    elif obv_code == "6299-2":
        # Urea Nitrogen
        return -100 # typical valid values are positive and > 0
    elif obv_code == "38483-4":
        # creatine
        return -100 # typical valid values are positive and > 0
    elif obv_code == "49765-1":
        # calcium
        return -100 # typical valid values are positive and > 0
    elif obv_code == "2947-0":
        # sodium
        return -100 # typical valid values are positive and > 0
    elif obv_code == "6298-4":
        # potassium
        return -100 # typical valid values are positive and > 0
    elif obv_code == "2069-3":
        # chloride
        return -100 # typical valid values are positive and > 0
    elif obv_code == "20565-8":
        # carbondioxide
        return -100 # typical valid values are positive and > 0
    elif obv_code == "33914-3":
        # Estimated Glomerular Filtration Rate
        return -100 # typical valid values are positve and > 0
    elif obv_code == "2885-2":
        # Protein [Mass/volume] in Serum or Plasma
        return -100 #typical valid values are positve and > 0
    elif obv_code == "1751-7":
        # Albumin [Mass/volume] in Serum or Plasma
        return -100 #typical valid values are positve and > 0
    elif obv_code == "10834-0":
        # Globulin [Mass/volume] in Serum by calculation
        return -100 # typical valid values are positve and > 0
    elif obv_code == "1975-2":
        # Bilirubin.total [Mass/volume] in Serum or Plasma
        return -100 # typical valid values are positve and > 0
    elif obv_code == "6768-6":
        # Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma
        return -100 # typical valid values are positve and > 0
    elif obv_code == "1742-6":
        # Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma
        return -100 # typical valid values are positve and > 0
    elif obv_code == "1920-8":
        # Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma
        return -100 # typical valid values are positve and > 0
    elif obv_code == "6690-2":
        # WBC Auto (Bld) [#/Vol]
        return -100 # typical valid values are positve and > 0
    elif obv_code == "789-8":
        # RBC Auto (Bld) [#/Vol]
        return -100 # typical valid values are positve and > 0
    elif obv_code == "718-7":
        # Hemoglobin [Mass/volume] in Blood
        return -100 # typical valid values are positve and > 0
    elif obv_code == "4544-3":
        # Hematocrit [Volume Fraction] of Blood by Automated count
        return -100 # typical valid values are positve and > 0
    elif obv_code == "787-2":
        # MCV [Entitic volume] by Automated count
        return -100 # typical valid values are positve and > 0
    elif obv_code == "785-6":
        # MCH [Entitic mass] by Automated count
        return -100 # typical valid values are positve and > 0
    elif obv_code == "786-4":
        # MCHC [Mass/volume] by Automated count
        return -100 # typical valid values are positve and > 0
    elif obv_code == "21000-5":
        # RDW - Erythrocyte distribution width Auto (RBC) [Entitic vol]
        return -100 # typical valid values are positve and > 0
    elif obv_code == "777-3":
        # Platelets [#/volume] in Blood by Automated count
        return -100 # typical valid values are positve and > 0
    elif obv_code == "32207-3":
        # Platelet distribution width [Entitic volume] in Blood by Automated count
        return -100 # typical valid values are positve and > 0
    elif obv_code == "32623-1":
        # Platelet mean volume [Entitic volume] in Blood by Automated count
        return -100 # typical valid values are positve and > 0
    elif obv_code == "72514-3":
        # Pain severity - 0-10 verbal numeric rating [Score] - Reported
        return -1 # typical valid values are positve between 0-9
    elif obv_code == "8331-1":
        # Oral temperature
        return -100 # typical valid values are positve and usually between 30 and 40
    elif obv_code == "33762-6":
        # NT-proBNP
        return -100 # typical valid values are positve and > 200
    elif obv_code == "8480-6":
        # Systolic Blood Pressure
        return -100 # typical valid values are positve and > 0
    elif obv_code == "8462-4":
        # Diastolic Blood Pressure
        return -100 # typical valid values are positve and > 0
    elif obv_code == "8462-4":
        # Diastolic Blood Pressure
        return -100 # typical valid values are positve and > 0
    elif obv_code == "2093-3":
        # Total Cholesterol
        return -100 # typical valid values are positve and > 0
    elif obv_code == "2571-8":
        # Triglycerides
        return -100 # typical valid values are positve and > 0
    elif obv_code == "18262-6":
        # Low Density Lipoprotein Cholesterol
        return -100 # typical valid values are positve and > 0
    elif obv_code == "2085-9":
        # High Density Lipoprotein Cholesterol
        return -100 # typical valid values are positve and > 0
    elif obv_code == "4548-4":
        # Hemoglobin A1c/Hemoglobin.total in Blood
        return -100 # typical valid values are positve and > 0
    elif obv_code == "19926-5":
        # FEV1/FVC
        return -100 # typical valid values are positve and > 0
    elif obv_code == "2857-1":
        # FEV1/FVC
        return -100 # typical valid values are positve and > 0
    elif obv_code == "8302-2":
        # Body Height
        return -100 # typical valid values are positve and > 0
    elif obv_code == "29463-7":
        # Body Weight
        return -100 # typical valid values are positve and > 0
    elif obv_code == "39156-5":
        # BMI ?? hmm
        return -100 # typical valid values are positve and > 0
    elif obv_code == "14959-1":
        # Microalbumin Creatinine Ratio
        return -100 # typical valid values are positve and > 0
    else:
        raise Exception("Invalid/Unsupported observation code %s" % obv_code)

def prep_data(csv_file):
    df = pd.read_csv(csv_file)
    to_drop = ['Unnamed: 0', 'marital_status', 'gender', '72166-2', 'race', 'ethnicity', 'encounter_id', 'patient_id', '59576-9', '77606-2']
    
    to_drop = list(set(df.columns) & set(to_drop))
    train_vector = df.drop(columns=to_drop)
    
    # missing data imputation?
    needs_filling = []
    for attr in train_vector.columns:
        num_na = train_vector[attr].isna().sum()
        if num_na > 0:
            needs_filling.append(attr)

    nan_fills = {attr: get_nan_value(attr) for attr in needs_filling}
    train_vector = train_vector.fillna(nan_fills)
    
    return train_vector