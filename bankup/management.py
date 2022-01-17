#import
import os
#Pandas
import pandas as pd
#pipeline
#package
import bankup.data as data


class Account(object):
    def __init__(self,Data_base):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X

    def import_new_data(self,X):
        """Import News values of data from account.
           For example: News data ==>new month, years of data.
        """
        X=data.get_data_clean(fichier,onglet)
        

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
        # path components
        path_v2 = ".."
        directory_name = "model"
        file_name='model_cost.joblib'
        file_name=os.path.join(path_v2,directory_name, file_name)
        
        joblib.dump(self.pipeline, file_name)
        print(f"saved {file_name} in directory: {directory_name}")
        
    def load_model(self):
        """method that load the model into a .joblib file """
        # load here
        pipeline = joblib.load('model_cost.joblib')
        print("model.joblib has been load ")
        return pipeline
