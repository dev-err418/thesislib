import json
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import StratifiedShuffleSplit


def extract_condition_symptom_from_modules(module_name):
    with open(module_name) as fp:
        module = json.load(fp)
    condition_code = None
    condition_name = None
    symptoms = []

    states = module.get('states')
    for state in states.values():
        state_type = state.get('type')
        if state_type == 'ConditionOnset':
            condition_code = state.get('codes')[0].get('code')
            condition_name = state.get('codes')[0].get('display')
        elif state_type == 'Symptom':
            symptom_code = state.get('symptom_code').get('code')
            symptoms.append(symptom_code)
    return condition_code, condition_name, set(symptoms)


def _race_txform(val):
    """
    Takes val which can either be white, black, asian, native or other
    And returns the integer code corresponding to the value
    :param val:
    :return:
    """
    race_code = {'white': 0, 'black':1, 'asian':2, 'native':3, 'other':4}
    return race_code.get(val)


def _label_txform(val, labels):
    """
    Given a mapping of values to integer code, this function returns the integer code corresponding to the provided
    val
    :param val:
    :param labels:
    :return:
    """
    return labels.get(val)


def _symptom_transform(val, labels):
    """
    Val is a string in the form: "symptom_0;symptom_1;...;symptom_n"
    This function assumes the coding scheme for the symptoms are chosen such the sum the transformed integer codes
    for any combination of symptoms is always unique.
    i.e if code_0, code_1, ..., code_n correspond to symptom_0, symptom_1, ..., symptom_n
    then the sum res = code_0 + code_1, ..., code_n can only be obtained from the exact combination of the symptoms.
    This allows using bitwise & operator to test if a code made up this sum.
    A classic example where this idea is used is in linux file system permissions where 1 corresponds to execute,
    2 corresponds to write and 4 corresponds to read. All possible sums in this combination of 1, 2, 4 are unique and given
    a sum value it is possible used bitwise & to determine which code made up the sum.

    This method allows for an easy to use method for combining symptoms.
    Though the performance has not been compared with converting the symptom string and doing

    :param val:
    :param labels:
    :return:
    """
    parts = val.split(";")
    res = sum([labels.get(item) for item in parts])
    return res


def handle_bit_wise(val, comp):
    if val & comp > 0:
        return 1
    else:
        return 0


def parse_data(filepath, condition_labels, symptom_map, output_path, use_header=True):

    symptom_columns = ['PATIENT', 'GENDER', 'RACE', 'ETHNICITY', 'AGE_BEGIN', 'AGE_END',
                       'PATHOLOGY', 'NUM_SYMPTOMS', 'SYMPTOMS']
    columns_interest = ['GENDER', 'RACE', 'AGE_BEGIN', 'PATHOLOGY', 'SYMPTOMS', 'NUM_SYMPTOMS']
    if use_header:
        symptoms_df = pd.read_csv(filepath, names=symptom_columns, usecols=columns_interest)
    else:
        symptoms_df = pd.read_csv(filepath, usecols=columns_interest)
    filename = filepath.split("/")[-1]

    # drop all extensions
    nameparts = filename.split(".")
    filename = nameparts[0] if len(nameparts) == 1 else "".join(nameparts[:-1])

    # drop the guys that have no symptoms
    symptoms_df = symptoms_df.loc[symptoms_df.NUM_SYMPTOMS > 0]

    symptoms_df['LABEL'] = symptoms_df.PATHOLOGY.apply(_label_txform, labels=condition_labels)
    symptoms_df.RACE = symptoms_df.RACE.apply(_race_txform)
    symptoms_df.GENDER = symptoms_df.GENDER.apply(lambda gender: 0 if gender == 'F' else 1)
    symptoms_df = symptoms_df.rename(columns={'AGE_BEGIN': 'AGE'})
    symptoms_df['NSYMPTOMS'] = symptoms_df.SYMPTOMS.apply(_symptom_transform, labels=symptom_map)

    vector = sorted(symptom_map.keys())
    for idx, code in enumerate(vector):
        symptoms_df[code] = (symptoms_df.NSYMPTOMS & 2 ** idx).gt(0).astype(np.uint8)

    ordered_keys = ['LABEL', 'GENDER', 'RACE', 'AGE'] + vector
    output_file = os.path.join(output_path, "%s.csv" % filename)
    symptoms_df = symptoms_df[ordered_keys]
    symptoms_df.to_csv(output_file)


def concatenate_and_split(filepath, output_path, train_split=0.8):
    filenames = sorted(glob(filepath))
    df = [pd.read_csv(file) for file in filenames]
    df = pd.concat(df)

    # now we have one big df
    labels = df.LABEL

    splitter = StratifiedShuffleSplit(1, train_size=train_split)
    train_index = None
    test_index = None
    for tr_idx, tst_index in splitter.split(df, labels):
        train_index = tr_idx
        test_index = tst_index
        break

    train_df = df.iloc[train_index]
    train_df.to_csv(os.path.join(output_path, "train.csv"))
    del train_df

    test_df = df.iloc[test_index]
    test_df.to_csv(os.path.join(output_path, "test.csv"))

    return True
