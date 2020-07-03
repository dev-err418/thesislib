import json
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pathlib
import hashlib
from glob import glob

from ..stringutils import  slugify

RACE_CODE = {'white': 0, 'black':1, 'asian':2, 'native':3, 'other':4}


def get_symptom_condition_map(module_dir):
    module_files = glob(os.path.join(module_dir, "*.json"))
    symptom_map = {}
    condition_map = {}
    for file in module_files:
        with open(file) as fp:
            module = json.load(fp)
        states = module.get("states")
        for state in states.values():
            if state.get("type") != "Symptom" and state.get("type") != "ConditionOnset":
                continue
            if state.get("type") == "ConditionOnset":
                code = state.get("codes")[0]
                condition_map[code["code"]] = slugify(code.get("display"))
                continue
            symptom_code = state.get("symptom_code")
            slug = slugify(symptom_code.get("display"))
            slug_hash  = hashlib.sha224(slug.encode("utf-8")).hexdigest()
            symptom_map[slug_hash] = slug
    return symptom_map, condition_map


def _symptom_transform(val, labels, is_nlice=False):
    """
    Val is a string in the form: "symptom_0;symptom_1;...;symptom_n"
    :param val:
    :param labels:
    :return:
    """
    parts = val.split(";")
    if is_nlice:
        indices = []
        for item in parts:
            id, enc = item.split("|")
            label = labels.get(id)
            indices.append("|".join([label, enc]))
        res = ",".join(indices)
    else:
        indices = []
        for item in parts:
            _ = labels.get(item)
            if _ is None:
                print(item)
                raise ValueError("Unknown symptom")
            indices.append(_)
        res = ",".join(indices)
    return res


def parse_data(
        filepath,
        conditions_db_json,
        symptoms_db_json,
        output_path,
        is_nlice=False,
        transform_map=None,
        encode_map=None,
        reduce_map=None):

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(symptoms_db_json) as fp:
        symptoms_db = json.load(fp)

    with open(conditions_db_json) as fp:
        conditions_db = json.load(fp)

    condition_labels = {code: idx for idx, code in enumerate(sorted(conditions_db.keys()))}
    symptom_map = {code: str(idx) for idx, code in enumerate(sorted(symptoms_db.keys()))}

    usecols = ['GENDER', 'RACE', 'AGE_BEGIN', 'PATHOLOGY', 'NUM_SYMPTOMS', 'SYMPTOMS']

    df = pd.read_csv(filepath, usecols=usecols)

    filename = filepath.split("/")[-1]

    # drop the guys that have no symptoms
    df = df[df.NUM_SYMPTOMS > 0]
    df['LABEL'] = df.PATHOLOGY.apply(lambda v: condition_labels.get(v))
    df['RACE'] = df.RACE.apply(lambda v: RACE_CODE.get(v))
    df['GENDER'] = df.GENDER.apply(lambda gender: 0 if gender == 'F' else 1)
    df = df.rename(columns={'AGE_BEGIN': 'AGE'})
    if is_nlice:
        df['SYMPTOMS'] = df.SYMPTOMS.apply(
            tranform_symptoms,
            transformation_map=transform_map,
            symptom_combination_encoding_map=encode_map,
            reduction_map=reduce_map)
    df['SYMPTOMS'] = df.SYMPTOMS.apply(_symptom_transform, labels=symptom_map, is_nlice=is_nlice)
    ordered_keys = ['LABEL', 'GENDER', 'RACE', 'AGE', 'SYMPTOMS']
    df = df[ordered_keys]
    df.index.name = "Index"

    output_file = os.path.join(output_path, "%s_sparse.csv" % filename)
    df.to_csv(output_file)

    return output_file


def split_data(symptom_file, output_path, use_headers=False, train_split=0.8):
    symptom_columns = ['PATIENT', 'GENDER', 'RACE', 'ETHNICITY', 'AGE_BEGIN', 'AGE_END',
                       'PATHOLOGY', 'NUM_SYMPTOMS', 'SYMPTOMS']

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    if use_headers:
        df = pd.read_csv(symptom_file, names=symptom_columns)
    else:
        df = pd.read_csv(symptom_file)

    df.index.name = "Index"

    labels = df["PATHOLOGY"]
    splitter = StratifiedShuffleSplit(1, train_size=train_split)
    train_index = None
    test_index = None
    for tr_idx, tst_index in splitter.split(df, labels):
        train_index = tr_idx
        test_index = tst_index
        break

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    train_op = os.path.join(output_path, "train.csv")
    test_op = os.path.join(output_path, "test.csv")
    train_df.to_csv(train_op)
    test_df.to_csv(test_op)
    return train_op, test_op


def tranform_symptoms(symptom_str, transformation_map, symptom_combination_encoding_map, reduction_map):
    symptom_list = symptom_str.split(";")
    symptoms = {}
    for item in symptom_list:
        transformed = transformation_map.get(item)
        if transformed is None:
            print(item)
            raise ValueError("found new symptom")
        name = transformed.get("symptom")
        if name not in symptoms:
            symptoms[name] = {
                "nature": "0-None",
                "vas": "0-None",
                "duration": "0-None",
                "location": "0-None"
            }
        nlice = transformed.get("nlice")
        nlice_value = transformed.get("value")
        if nlice is not None and nlice_value is not None:
            if name in reduction_map and nlice_value in reduction_map[name]:
                nlice_value = reduction_map[name][nlice_value]
            symptoms[name][nlice] = nlice_value

    transformed_symptoms = []
    for key, value in symptoms.items():
        ordered = [value.get(item) for item in ["nature", "vas", "duration", "location"]]
        ordered = ";".join(ordered)
        encoding = symptom_combination_encoding_map[key][ordered]
        symptom_hash = hashlib.sha224(key.encode("utf-8")).hexdigest()
        transformed_symptoms.append("|".join([symptom_hash, str(encoding)]))
    return ";".join(transformed_symptoms)
