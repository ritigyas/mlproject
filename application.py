from flask import Flask, request, render_template  # type: ignore[import]
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

LESSONS = {
    "problem-statement": {
        "number": "01", "icon": "bi-bullseye", "title": "Problem statement discussion",
        "summary": "Define a useful regression problem before writing model code.",
        "why": "The application predicts a student's mathematics score from contextual and assessment features. The target is numeric, so this is a supervised regression task.",
        "steps": ["Identify math_score as the prediction target.", "Choose meaningful inputs: demographic context, lunch, preparation, reading, and writing scores.", "Decide how success will be measured with an R² score on unseen data."],
        "source_file": "notebook/eda_student.ipynb",
    },
    "data-ingestion": {
        "number": "02", "icon": "bi-database-down", "title": "Data ingestion implementation",
        "summary": "Load the source dataset, preserve a raw copy, and create reproducible train/test data.",
        "why": "Data ingestion makes the first step repeatable. It reads the student dataset, writes a raw artifact, and creates an 80/20 split with random_state=42.",
        "steps": ["Read notebook/data/stud.csv with pandas.", "Save a raw CSV copy in artifacts/raw.csv.", "Split the dataset into train.csv and test.csv with train_test_split."],
        "source_file": "src/components/data_ingestion.py",
    },
    "data-transformation": {
        "number": "03", "icon": "bi-funnel", "title": "Data transformation using pipelines",
        "summary": "Prepare numeric and categorical features without leaking information from test data.",
        "why": "A ColumnTransformer keeps each feature type on the correct preprocessing path and saves the fitted preprocessor for inference.",
        "steps": ["Median-impute and scale reading and writing scores.", "Most-frequent-impute and one-hot encode categorical fields.", "Fit preprocessing on training data only, then transform testing data and save preprocessor.pkl."],
        "source_file": "src/components/data_transformation.py",
    },
    "model-training": {
        "number": "04", "icon": "bi-cpu", "title": "Model trainer implementation",
        "summary": "Train multiple candidate regressors and persist the best-performing model.",
        "why": "Comparing models makes the selection evidence-based instead of assuming one algorithm will work best for every dataset.",
        "steps": ["Train Linear Regression, tree ensembles, boosting methods, KNN, XGBoost, and CatBoost.", "Evaluate candidates through the reusable evaluate_models utility.", "Save the selected model to artifacts/model.pkl when it clears the quality threshold."],
        "source_file": "src/components/model_trainer.py",
    },
    "hyperparameter-tuning": {
        "number": "05", "icon": "bi-sliders2", "title": "Hyperparameter tuning",
        "summary": "Search useful model configurations to improve performance beyond defaults.",
        "why": "Hyperparameters control model capacity and learning behavior; tuning helps balance bias and variance using validation evidence.",
        "steps": ["Define candidate values such as estimators, learning rate, depth, and neighbors.", "Run the search inside evaluate_models for each supported model.", "Compare the resulting R² scores and retain the strongest candidate."],
        "source_file": "src/components/model_trainer.py",
    },
    "prediction-pipeline": {
        "number": "06", "icon": "bi-lightning-charge", "title": "Building the prediction pipeline",
        "summary": "Apply the exact training-time preprocessing before generating a prediction.",
        "why": "Serving must use the saved preprocessor and saved model together so the feature representation stays consistent with training.",
        "steps": ["Collect form fields with the CustomData helper.", "Convert one student record into a pandas DataFrame.", "Load model.pkl and preprocessor.pkl, transform inputs, then return the estimated score."],
        "source_file": "src/pipeline/predict_pipeline.py",
    },
    "deployment": {
        "number": "07", "icon": "bi-github", "title": "Deployment, GitHub & code setup",
        "summary": "Expose the trained pipeline through a small Flask application and a clear project structure.",
        "why": "The UI is the final delivery layer: it lets a user provide data, triggers inference, and presents the result in a usable form.",
        "steps": ["Use Flask routes for the overview, learning modules, and prediction form.", "Keep components, pipelines, artifacts, and templates in focused folders.", "Publish the project code and documentation to GitHub for review and reuse."],
        "source_file": "app.py",
    },
}

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/learn/<lesson_slug>')
def learn(lesson_slug):
    lesson = LESSONS.get(lesson_slug)
    if lesson is None:
        return render_template('404.html'), 404
    lesson["github_url"] = "https://github.com/ritigyas/mlproject/blob/main/" + lesson["source_file"]
    return render_template('lesson.html', lesson=lesson)

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html', results=None)
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

