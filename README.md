# Disaster Response Pipeline Project

### Installation:
Python 3 is used for coding. You will need to the following python packages installed
for running the ML pipelines:
1. SciKit-Learn
2. Pandas
3. NLTK

In addition, SQLAlchemy, SQLite, and Flask are also required for the web app and database management.

### Motivation
Immediately following an extreme event like flooding or a hurricane, there is a deluge of messages for requesting assistance, informing people impacted, etc. For responders and relief organizations, it is vital to sort all these communications and extract the most informative/helpful ones.

We develop a machine learning pipeline for cleaning and classifying the communications using Natual Language Processing techniques. The model is then deployed in a web app to classify new messages in real time.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Results
The trained pipeline has an average accuracy of 85% and is capable of predicting the most likely/relevant context for the message out of the 36 available categories.

### Acknowledgements
Data is provided by Figure8. Web app template is mostly done by Udacity.