#import
import os
from pathlib import Path
#Pandas
import pandas as pd
#pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
#ohe
from sklearn.preprocessing import OneHotEncoder
#Extraction text
from sklearn.feature_extraction.text import CountVectorizer
#model
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
import joblib
#train
from sklearn.model_selection import train_test_split
#package
import bankup.data as data
from bankup.encoder_v1 import CyclicalEncoder


class Trainer_cost_model(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        #Def path to stock and run model.joblibs
        self.model_directory= "model"
        self.model_file='model_cost.joblib'
        path=Path(__file__).parents[0]
        self.path=os.path.join(path,self.model_directory)
        self.file_name=os.path.join(self.path, self.model_file)


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        preproc_pipe = ColumnTransformer([
            ('time_cyclical', CyclicalEncoder('month'),
                        ['month']),
            ('ohe_date_charge_groupe', OneHotEncoder(handle_unknown='ignore',sparse=False),
                        ["year","month",
                        #"Type de charge ou revenus",
                        "Groupes par structure"])
                                        ], remainder="drop")

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model',  Lasso())
            ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """return dataframe with X_test, ytest and y_prediction"""
        y_pred = self.pipeline.predict(X_test)
        result_df=pd.concat([
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            pd.DataFrame(y_pred, columns=["y_pred"])
          ],axis=1)
        return result_df

    def score(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the r2"""
        # Score model
        r2 = self.pipeline.score(X_test,y_test)
        return round(r2, 2)

    def save_model(self):
        """method that saves the model into a .joblib file """       
        joblib.dump(self.pipeline, self.file_name)
        print(f"saved {self.model_file} in directory: {self.path}")
        
    def load_model(self):
        """method that load the model into a .joblib file """
        # load here
        pipeline = joblib.load(self.file_name)
        print(f"{self.model_file} has been load ")
        return pipeline

class Trainer_class_model_label(object):
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = None
        self.y = None
        #Def path to stock and run model.joblibs
        self.model_directory= "model"
        self.model_file='model_classification_label.joblib'
        
        path_local=Path(__file__).parents[0]
        #Path du pakage quand c'est pakagé==> a utilisé pour la fonction "load_model"
        path_package=Path(__file__).parents[0]
        self.path=os.path.join(path,self.model_directory)
        self.file_name=os.path.join(self.path, self.model_file)

    def set_pipeline(self,list_stop_word=None):
        """defines the pipeline as a class attribute"""
        Clean_text = Pipeline([
                ('clean_text', FunctionTransformer(data.clean_data_text))   
                        ])
        tree = DecisionTreeClassifier()

        self.pipeline = Pipeline([
                ('clean_text', Clean_text),
                ('Bag of word',CountVectorizer(stop_words=list_stop_word)),
                ('tree_classifier',tree)     
                        ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """return dataframe with X_test, ytest and y_prediction"""
        y_pred = self.pipeline.predict(X_test)
        result_df=pd.concat([
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            pd.DataFrame(y_pred, columns=["y_pred"])
          ],axis=1)
        return result_df

    def score(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the r2"""
        # Score model
        r2 = self.pipeline.score(X_test,y_test)
        return round(r2, 2)
 
    def save_model(self):
        """method that saves the model into a .joblib file """       
        joblib.dump(self.pipeline, self.file_name)
        print(f"saved {self.model_file} in directory: {self.path}")
        
    def load_model(self):
        """method that load the model into a .joblib file """
        # load here
        pipeline = joblib.load(self.file_name)
        print(f"{self.model_file} has been load ")
        return pipeline

class Trainer_class_model_enseigne(object):
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = None
        self.y = None
        #Def path to stock and run model.joblibs
        self.model_directory= "model"
        self.model_file='model_classification_enseigne.joblib'
        path=Path(__file__).parents[0]
        self.path=os.path.join(path,self.model_directory)
        self.file_name=os.path.join(self.path, self.model_file)

    def set_pipeline(self,list_stop_word=None):
        """defines the pipeline as a class attribute"""
        Clean_text = Pipeline([
                ('clean_text', FunctionTransformer(data.clean_data_text))   
                        ])
        tree = DecisionTreeClassifier()

        self.pipeline = Pipeline([
                ('clean_text', Clean_text),
                ('Bag of word',CountVectorizer(stop_words=list_stop_word)),
                ('tree_classifier',tree)     
                        ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """return dataframe with X_test, ytest and y_prediction"""
        y_pred = self.pipeline.predict(X_test)
        result_df=pd.concat([
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            pd.DataFrame(y_pred, columns=["y_pred"])
          ],axis=1)
        return result_df

    def score(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the r2"""
        # Score model
        r2 = self.pipeline.score(X_test,y_test)
        return round(r2, 2)

    def save_model(self):
        """method that saves the model into a .joblib file """       
        joblib.dump(self.pipeline, self.file_name)
        print(f"saved {self.model_file} in directory: {self.path}")
        
    def load_model(self):
        """method that load the model into a .joblib file """
        # load here
        pipeline = joblib.load(self.file_name)
        print(f"{self.model_file} has been load ")
        return pipeline


if __name__ == "__main__":
    
    
    test=Path(__file__).parents[0]
    # Get and clean data
    fichier='releves_banque_2015_16_17_18_19_20_21_V7.xls'
    onglet='Releve_compte'
    df = data.get_data_cost_by_month(fichier,onglet)
    print(df)
    X=df.drop(columns=["Valeurs"])
    y = df["Valeurs"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
     # Train and save model, locally and
    trainer = Trainer_cost_model(X=X_train, y=y_train)
    trainer.run()
    r2 = trainer.score(X_test, y_test)
    print(f"r2: {r2}")
    predict_df = trainer.evaluate(X_test, y_test)
    print(predict_df)
    trainer.save_model()
