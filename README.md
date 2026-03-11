Dataset setup

1. Download the AEON essays dataset from Kaggle:
https://www.kaggle.com/datasets/mannacharya/aeon-essays-dataset

2. Create the following folders:

textdetector/data/raw

textdetector/data/processed

3. Place the downloaded dataset in:
textdetector/data/raw/essays.csv

4. Process the human essays:
python data/human_essays.py


5. Generate AI essays:
python data/generate_ai_essays.py
