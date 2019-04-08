# Description
This project is aimed at predicting age of death based on mortality dataset from the CDC found [here](https://www.kaggle.com/cdc/mortality). The original dataset has been preprocessed to remove unnecessary fields, and the actual age of death information. The remaining columns used in training are the following:
* resident_status
* education
* marital_status
* race
* sex
* hispanic_origin
* 20x conditions

from these columns we aim to predict age range of the time of death. We use past years dataset for training and use the following year for validation. 

# Getting Started

To run the program `py main.py`
