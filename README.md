# MVR-GA
The source code of paper, “A Multi-view Learning-Based Rule Extraction Algorithm For Accurate Hepatotoxicity Prediction”.

## Requirements
The requirements of MVR-GA can see the requirements.txt.
```bash
Python==3.8
```

# Usage

## Dataset Preparation
MVR-GA adopts the 5-Fold Cross-Validation.

You may run the datacross.py in folder data. And then copy the csv files in one of the folder dataCrossResult to the folder dataset.

```bash
cd data
python datacross.py
```

## Training the MVR-GA
You may take the following two steps. 

Firstly, run the mainRule.py. Four rule.py files will be generated. These are the four rulesets used for GA optimization on each view.

```bash
python mainRule.py
```

Secondly, run the mainGa.py to perform the GA optimization. 

```bash
python mainGa.py
```

## Get results

The accuracy and coverage of MVR-GA can be seen in the acc_after_ga.csv file in the results folder. 

The details of rules can be seen in the folder ruleIndex and folder ruleInformation.
