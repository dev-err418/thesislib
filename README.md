## Description

This python library contains a collection of useful classes/methods related to my thesis project: 
[Effective Primary Healthcare Differential Diagnosis: A Machine Learning Approach](https://repository.tudelft.nl/islandora/object/uuid%3A5b9d542f-7d61-4c11-9646-474be1b85fca?collection=education)

The library is organised as follows:
```bash
├── __init__.py
└── utils
    ├── __init__.py
    ├── dl
    │   ├── __init__.py
    │   ├── bench.py
    │   ├── dae.py
    │   ├── data.py
    │   ├── models.py
    │   ├── runners.py
    │   └── utils.py
    ├── imput
    │   ├── __init__.py
    │   ├── imput.py
    │   └── utils.py
    ├── knifeutils.py
    ├── ml
    │   ├── __init__.py
    │   ├── models.py
    │   ├── process.py
    │   ├── report.py
    │   └── runners.py
    ├── pathutils.py
    ├── rl
    │   ├── __init__.py
    │   ├── agent.py
    │   ├── bench.py
    │   └── environment.py
    ├── statutils.py
    └── stringutils.py

```

The `thesislib.utils.dl` module contains code related to exploration of deep learning techniques - specifically a fully
connected neural network and an attempt at using a deep auto encoder for dimensionality reduction.

The `thesislib.utils.imput` module is a vestige from the beginning of the project where the data source at the time
would have required the utilization of imputation techniques.

The `thesislib.utils.knifeutils` contains helper code for data processing that has been moved to the `thesislib.utils.ml`
module.

The `thesislib.utils.ml` contains the core part of the code. It contains a Naive Bayes model customized to suit the 
data source. It also contains helper code for processing the data, specifying configuration options (e.g for the Random
Forest classifier) and model evaluation functions.

The `thesislib.utils.pathutils` is also a vestige from before the code was packaged as a library. It allowed the use of 
relative paths when accessing data files.

The `thesislib.utils.rl` module contains a base implementation of a reinforced learning (RL) approach to automate symptom
discovery/acquisition from the patient and eventual differential diagnosis. It contains an implementation of the 
patient (or environment in RL speak), and the doctor (agent in RL speak which learns to ask questions and make diagnosis).
It uses a Deep-Q network in training the agent.

The `thesislib.utils.statutils` and `thesislib.utils.stringutils` both contain helper functions.

## Installation

You can install this library by using:
```bash
python -m pip install git+https://github.com/teliov/thesislib.git@{{commit-hash}}
```

If you're installing via a `requirements.txt` file then add this line to the file:
```bash
-e git+https://github.com/teliov/thesislib.git@{{commit-hash}}#egg=thesislib
```