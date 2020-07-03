import json
import os
from collections import namedtuple
from itertools import repeat
from scipy.sparse import  csc_matrix

import numpy as np

AiMedPatient = namedtuple('AiMedPatient', ('age', 'race', 'gender', 'symptoms', 'condition'))
AiMedState = namedtuple('AiMedState', ('age', 'race', 'gender', 'symptoms'))


class AiBasicMedEnv:
    def __init__(
            self,
            data_file,
            symptom_map_file,
            condition_map_file,
            clf
    ):
        """
        data_file: A file of generated patient, symptoms, condition data
        symptom_map_file: the encoding file for symptoms
        condition_map_file: the encoding file for conditions
        initial_symptom_file: a map of conditions
        clf: a classifier which can output a probabilistic description of possible conditions based on
        symptoms and patient demography.
        """
        self.data_file = data_file
        self.symptom_map_file = symptom_map_file
        self.condition_map_file = condition_map_file
        self.clf = clf

        self.line_number = 0
        self.state = None
        self.patient = None
        self.data = None
        self.symptom_map = None
        self.condition_map = None
        self.initial_symptom_map = None
        self.num_symptoms = None
        self.num_conditions = None

        self.check_file_exists()

        self.load_data_file()
        self.load_symptom_map()
        self.load_condition_map()

        self.is_inquiry = 1
        self.is_diagnose = 2

        self.inquiry_list = set([])

        self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}

    def check_file_exists(self):
        files = [self.data_file, self.symptom_map_file, self.condition_map_file]
        for file in files:
            if not os.path.exists(file):
                raise ValueError("File: %s does not exist" % file)

    def load_data_file(self):
        self.data = open(self.data_file)

    def close_data_file(self):
        if self.data is not None:
            self.data.close()

    def load_symptom_map(self):
        with open(self.symptom_map_file) as fp:
            symptoms = json.load(fp)
            sorted_symptoms = sorted(symptoms.keys())
            self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
            self.num_symptoms = len(self.symptom_map)

    def load_condition_map(self):
        with open(self.condition_map_file) as fp:
            conditions = json.load(fp)
            sorted_conditions = sorted(conditions.keys())
            self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
            self.num_conditions = len(self.condition_map)

    def readline(self):
        line = self.data.readline()
        line = "" if line is None else line.strip()
        return line

    def get_line(self):
        if self.line_number == 0:
            self.readline() # header line

        line = self.readline()
        if not line or len(line) == 0:
            # EOF
            self.data.seek(0)
            self.readline() # header line
            line = self.readline()

        self.line_number += 1
        return line

    def parse_line(self, line):
        parts = line.split(",")
        _gender = parts[1]
        _race = parts[2]

        age = int(parts[4])
        condition = parts[6]
        symptom_list = parts[8]

        gender = 0 if _gender == 'M' else 1
        race = self.RACE_CODE.get(_race)
        condition = self.condition_map.get(condition)
        symptoms = list(repeat(0, self.num_symptoms))
        for item in symptom_list.split(";"):
            idx = self.symptom_map.get(item)
            symptoms[idx] = 1
        # ('age', 'race', 'gender', 'symptoms', 'condition')
        symptoms = np.array(symptoms)
        patient = AiMedPatient(age, race, gender, symptoms, condition)
        return patient

    def reset(self):
        line = self.get_line()
        self.patient = self.parse_line(line)
        self.state = self.generate_state(
            self.patient.age,
            self.patient.race,
            self.patient.gender
        )
        self.inquiry_list = set([])

        self.pick_initial_symptom()

    def pick_initial_symptom(self):
        _existing_symptoms = np.where(self.patient.symptoms == 1)[0]

        initial_symptom = np.random.choice(_existing_symptoms)

        self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
        self.inquiry_list.add(initial_symptom)

    def generate_state(self, age, race, gender):
        _symptoms = np.zeros((self.num_symptoms, 3), dtype=np.uint8)  # all symptoms start as unknown
        _symptoms[:, 2] = 1

        return AiMedState(age, race, gender, _symptoms)

    def is_valid_action(self, action):
        if action < self.num_symptoms:
            return True, self.is_inquiry, action  # it's an inquiry action
        else:
            action = action % self.num_symptoms

            if action < self.num_conditions:
                return True, self.is_diagnose, action  # it's a diagnose action

        return False, None, None

    def take_action(self, action):
        is_valid, action_type, action_value = self.is_valid_action(action)
        if not is_valid:
            raise ValueError("Invalid action: %s" % action)
        if action_type == self.is_inquiry:
            return self.inquire(action_value)
        else:
            return self.diagnose(action_value)

    def patient_has_symptom(self, symptom_idx):
        return self.patient.symptoms[symptom_idx] == 1

    def inquire(self, action_value):
        """
        returns state, reward, done
        """
        if action_value in self.inquiry_list:
            # repeated inquiry
            # return self.state, -1, True # reward is -3 if inquiry is repeated
            return self.state, -3, False

        # does the patient have the symptom
        if self.patient_has_symptom(action_value):
            value = np.array([0, 1, 0])
        else:
            value = np.array([1, 0, 0])

        self.state.symptoms[action_value] = value
        self.inquiry_list.add(action_value)

        return self.state, -1, False # reward is -1 for non-repeated inquiry

    def get_patient_vector(self):
        patient_vector = np.zeros(3 + self.num_symptoms, dtype=np.uint8)
        patient_vector[0] = self.state.gender
        patient_vector[1] = self.state.race
        patient_vector[2] = self.state.age

        has_symptom = np.where(self.state.symptoms[:, 1] == 1)[0] + 3
        patient_vector[has_symptom] = 1

        return patient_vector.reshape(1, -1)

    def predict_condition(self):
        patient_vector = self.get_patient_vector()
        patient_vector = csc_matrix(patient_vector)

        prediction = self.clf.predict(patient_vector)

        return prediction

    def diagnose(self, action_value):
        # enforce that there should be at least one inquiry in addition to the initial symptom
        if len(self.inquiry_list) < 2:
            return self.state, -3, False # ask at least one question before attempting to diagnose

        # we'll need to make a prediction
        prediction = self.predict_condition()[0]

        is_correct = action_value == prediction
        reward = 1 if is_correct else 0

        return None, reward, True

    def __del__(self):
        self.close_data_file()


class AiBasicMedEnvSample:
    def __init__(
            self,
            definition_file,
            symptom_map_file,
            condition_map_file,
            clf
    ):
        """
        data_file: A file of generated patient, symptoms, condition data
        symptom_map_file: the encoding file for symptoms
        condition_map_file: the encoding file for conditions
        initial_symptom_file: a map of conditions
        clf: a classifier which can output a probabilistic description of possible conditions based on
        symptoms and patient demography.
        """
        self.symptom_map_file = symptom_map_file
        self.condition_map_file = condition_map_file
        self.definition_file = definition_file
        self.clf = clf

        self.state = None
        self.patient = None
        self.symptom_map = None
        self.condition_map = None
        self.definition = None
        self.initial_symptom_map = None
        self.num_symptoms = None
        self.num_conditions = None

        self.load_definition_file()
        self.load_symptom_map()
        self.load_condition_map()

        self.is_inquiry = 1
        self.is_diagnose = 2

        self.inquiry_list = set([])

        self.RACE_CODE = {'white': 0, 'black': 1, 'asian': 2, 'native': 3, 'other': 4}

    def load_symptom_map(self):
        with open(self.symptom_map_file) as fp:
            symptoms = json.load(fp)
            sorted_symptoms = sorted(symptoms.keys())
            self.symptom_map = {code: idx for idx, code in enumerate(sorted_symptoms)}
            self.num_symptoms = len(self.symptom_map)

    def load_condition_map(self):
        with open(self.condition_map_file) as fp:
            conditions = json.load(fp)
            sorted_conditions = sorted(conditions.keys())
            self.condition_map = {code: idx for idx, code in enumerate(sorted_conditions)}
            self.num_conditions = len(self.condition_map)

    def load_definition_file(self):
        with open(self.definition_file) as fp:
            self.definition = json.load(fp)

    def generate_patient(self):
        return AiMedPatient(-1, -1, -1, -1, -1)

    def reset(self):
        self.patient = self.generate_patient()
        self.state = self.generate_state(
            self.patient.age,
            self.patient.race,
            self.patient.gender
        )
        self.inquiry_list = set([])

        self.pick_initial_symptom()

    def pick_initial_symptom(self):
        _existing_symptoms = np.where(self.patient.symptoms == 1)[0]

        initial_symptom = np.random.choice(_existing_symptoms)

        self.state.symptoms[initial_symptom] = np.array([0, 1, 0])
        self.inquiry_list.add(initial_symptom)

    def generate_state(self, age, race, gender):
        _symptoms = np.zeros((self.num_symptoms, 3), dtype=np.uint8)  # all symptoms start as unknown
        _symptoms[:, 2] = 1

        return AiMedState(age, race, gender, _symptoms)

    def is_valid_action(self, action):
        if action < self.num_symptoms:
            return True, self.is_inquiry, action  # it's an inquiry action
        else:
            action = action % self.num_symptoms

            if action < self.num_conditions:
                return True, self.is_diagnose, action  # it's a diagnose action

        return False, None, None

    def take_action(self, action):
        is_valid, action_type, action_value = self.is_valid_action(action)
        if not is_valid:
            raise ValueError("Invalid action: %s" % action)
        if action_type == self.is_inquiry:
            return self.inquire(action_value)
        else:
            return self.diagnose(action_value)

    def patient_has_symptom(self, symptom_idx):
        return self.patient.symptoms[symptom_idx] == 1

    def inquire(self, action_value):
        """
        returns state, reward, done
        """
        if action_value in self.inquiry_list:
            # repeated inquiry
            # return self.state, -1, True  # we terminate on a repeated inquiry
            return None, -1, True

        # does the patient have the symptom
        if self.patient_has_symptom(action_value):
            value = np.array([0, 1, 0])
        else:
            value = np.array([1, 0, 0])

        self.state.symptoms[action_value] = value
        self.inquiry_list.add(action_value)

        return self.state, 0, False

    def get_patient_vector(self):
        patient_vector = np.zeros(3 + self.num_symptoms, dtype=np.uint8)
        patient_vector[0] = self.state.gender
        patient_vector[1] = self.state.race
        patient_vector[2] = self.state.age

        has_symptom = np.where(self.state.symptoms[:, 1] == 1)[0] + 3
        patient_vector[has_symptom] = 1

        return patient_vector.reshape(1, -1)

    def predict_condition(self):
        patient_vector = self.get_patient_vector()
        patient_vector = csc_matrix(patient_vector)

        prediction = self.clf.predict(patient_vector)

        return prediction

    def diagnose(self, action_value):
        # enforce that there should be at least one inquiry in addition to the initial symptom
        if len(self.inquiry_list) < 2:
            # return self.state, -1, True  # we always terminate on a repeated enquiry
            return None, -1, True

        # we'll need to make a prediction
        prediction = self.predict_condition()[0]

        is_correct = action_value == prediction
        reward = 1 if is_correct else 0

        return None, reward, True
