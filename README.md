Dataset setup

Download the AEON essays dataset from Kaggle:
https://www.kaggle.com/datasets/mannacharya/aeon-essays-dataset

Create the following folders:

textdetector/data/raw
textdetector/data/processed

Place the downloaded dataset in:
textdetector/data/raw/essays.csv

Process the human essays:
python data/human_essays.py


Generate AI essays:
python data/generate_ai_essays.py
