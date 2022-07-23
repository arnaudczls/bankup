#import
import os
#Pandas
import pandas as pd
#json
import json
#Path
from pathlib import Path
#pipeline
#package
import bankup.data as data
from bankup.trainer_v2 import  Trainer_class_model_label,Trainer_class_model_enseigne


class Account(object):
    def __init__(self):
        """
            class which can add new bank statement.
            Input: file.csv
            Output: dataframe with 5 columns
                    1==>Label (from file.csv)
                        ex:CB AUCHAN DRIVE FACT 190321
                    2==>enseigne predicted
                        ex:AUCHAN
                    3==>type predicted
                        ex:Alimentation domicile
                    4==>structure
                        ex:Alimentation
                    5==>charge
                        ex:Charges courantes
                        
            Posibility to calculate balance of statement by month.
            Calculation of each cost and income to see at the end of the month, there are more cost than income.
            function==> get_balance_cost_income_by_month
        """
        self.categorisation_definition_df = None
        #self.X = X
        
    def categorisation_definition(self):
        """Import dataframe with the architecture of categorisation.
        """
        
        #get path and name file.json
        fichier='categorisation_table.json'
        #Path du pakage quand c'est pakagé==> a utilisé pour la fonction "load_model"
        path_package=Path(__file__).parents[0]
        fichier=os.path.join(path_package,fichier)
        
        with open(fichier) as f:
            dico = json.load(f)
        f.close()

        #Structure_df=pd.DataFrame.from_dict(dico,orient="index")

        Groupes_par_structure=[]
        for key in dico.keys():
            #print(key)
            tmp=[key for i in range(len(dico[key]))]
            Groupes_par_structure.extend(tmp)
            
        Groupes_par_type=[]
        for value in dico.values():
            Groupes_par_type.extend(value)
    
        self.categorisation_definition_df=pd.concat([
            pd.DataFrame(Groupes_par_structure,columns=["Groupes par structure"]),
            pd.DataFrame(Groupes_par_type,columns=["Groupes par type"])
                ],axis=1)
        self.categorisation_definition_df["Type de charge ou revenus"]=self.categorisation_definition_df["Groupes par structure"].apply(Account.categorisation_revenus_charge)
        return self.categorisation_definition_df
    
    def categorisation_revenus_charge(x):
        #Function perform to clasifi charges
        switcher={
            "Revenus":"Revenus",
            "Revenus du capital":"Revenus du capital",
            "Habitation":"Charges fixes",
            "Assurance":"Charges fixes",  
            "Garde enfants":"Charges fixes",
            "Scolarité":"Charges fixes",
            "Impot":"Charges fixes",
            "Alimentation":"Charges courantes",
            "Entretien du logement":"Charges courantes",
            "Amenagement du logement": "Charges occasionnelles",
            "Equipement du logement":"Charges expectionnelles",
            "Transport":"Charges courantes",
            "Vehicule":"Charges occasionnelles",
            "Achat vehicule":"Charges expectionnelles",
            "Habillement":"Charges courantes",
            "Beaute et bien etre":"Charges courantes",
            "Sante":"Charges courantes",
            "Culture, loisirs, education et sport":"Charges courantes",
            "Moka":"Charges courantes",
            "Vacances":"Charges expectionnelles",
            "Epargne cour terme":"Epargne",    
            "Epargne long terme":"Charges fixes",
            "Epargne moyen terme":"Epargne",
            "Divers":"Charges occasionnelles",
            "Service":"Charges fixes",
            "Poste":"Charges occasionnelles",
            "Frais professionnel":"Charges occasionnelles",
            "Amende":"Charges expectionnelles",
            "Evenement de vie":"Charges expectionnelles",
        }
        return switcher.get(x,"not find")
    
    def categorize_new_bank_statement(self,Bank_statement_csv_name):
        """1) get label on bank statement from csv.
           2) run ML classification to get from label, the enseigne ==> New column "Groupes par enseigne"
           3) run ML classification to get from the enseigne, the type ==> New column "Groupes par type"
           4) perform a loop for to categorize from the type to structure ==> New column "Groupes par structure"
              perform a loop for to categorize from the type to structure ==> New column "Type de charge ou revenus"

           input:
           Bank_statement_csv_name (string)==> the name of csv
           
        """
        #1) get label on bank statement from csv.
        df=pd.read_csv(Bank_statement_csv_name,delimiter=";",encoding="cp1252",header=None)
        df=df.iloc[:,1:2]
        df=df.dropna()
        df=df.reset_index(drop=True)
        df=df.rename(columns={1: "Label"})

        #2) run ML classification to get from label, the enseigne ==> New column "Groupes par enseigne
        #import ML classification model
        class_model_enseigne=Trainer_class_model_label()
        pipeline_enseigne=class_model_enseigne.load_model_package()
        print("ML classification model to get from label, the enseigne",pipeline_enseigne)
        # make prediction
        results = pipeline_enseigne.predict(df.Label)
        
        #join result in the same df
        results_df=pd.concat([
                    df,
                    pd.DataFrame(results,columns=["enseigne predicted"]),
                ],axis=1)

        #3) run ML classification to get from the enseigne, the type ==> New column "Groupes par type"
        #import ML classification model
        class_model_type=Trainer_class_model_enseigne()
        pipeline_type=class_model_type.load_model_package()
        print("ML classification model to get from the enseigne,the type",pipeline_type)
        # make prediction
        results = pipeline_type.predict(results_df["enseigne predicted"])
        
        #join result in the same df
        results_df=pd.concat([
                    results_df,
                    pd.DataFrame(results,columns=["type predicted"]),
                ],axis=1)

        #4) perform a loop for to categorize from the type to structure ==> New column "Groupes par structure"
        categorisation_df=self.categorisation_definition()
        #get the nombre of unique type from
        list_unique=results_df["type predicted"].unique()
        
        #Initialisation of new column
        results_df["structure"]=""
        df=categorisation_df
        for i in list(list_unique):
            #Find all type (for exemple "impot") in dataframe test
            structure_tmp=categorisation_df[
                                categorisation_df[["Groupes par type"]].eq(i).any(1)
                                            ][["Groupes par structure","Type de charge ou revenus"]]
            if structure_tmp.empty:
                structure_tmp=pd.DataFrame()
                structure_tmp["Groupes par structure"]=["not find"]
                structure_tmp["Type de charge ou revenus"]=["not find"]
            #write in column "structure" the structure with the method loc
            results_df.loc[results_df["type predicted"].eq(i),"structure"]=list(structure_tmp["Groupes par structure"])[0]
            #write in column "Type de charge ou revenus" the structure with the method loc
            results_df.loc[results_df["type predicted"].eq(i),"charge"]=list(structure_tmp["Type de charge ou revenus"])[0] 

        return results_df

    def get_balance_cost_income_by_month (releve_compte_month):
        #----------info---------------
        #Posibility to calculate balance of statement by month.
        #Calculation of each cost and income to see at the end of the month, there are more cost than income.
        # 
        #The summe of income and cost is done each month   
        #    
        #Warning!!!!
        #the caterogry Epargne and Revenus du capital are exlut for the calculation
        #
        #Input:
        #   releve_compte_month=dataframe which comme from function data.get_data_by_month
        #
        #Output:
        #dataframe
        #
        #Mise a jour: 09/07/2022
        #----------end info---------------
        
        #Initialisation
        balance_df=pd.DataFrame(columns=["year","month","benefice lost"])
        year_i=releve_compte_month.iloc[0]["year"]
        month_i=releve_compte_month.iloc[0]["month"]
        somme=0
        
        #loop
        for i in range(len(releve_compte_month)):
            #years
            if releve_compte_month.iloc[i]["year"]==year_i:
                #month
                if releve_compte_month.iloc[i]["month"]==month_i:
                    test=releve_compte_month.iloc[i]["Type de charge ou revenus"]
                    if (test!='Epargne'and test!='Revenus du capital'):
                        somme=somme+releve_compte_month.iloc[i]["Valeurs"] 
                # write df and change month
                else:
                    #write df        
                    balance_df.loc[-1] = [year_i, month_i, somme]
                    balance_df.index = balance_df.index + 1
                    balance_df = balance_df.sort_index()
                    #change month
                    month_i=releve_compte_month.iloc[i]["month"]
                    somme=0
                    test=releve_compte_month.iloc[i]["Type de charge ou revenus"]
                    if (test!='Epargne'and test!='Revenus du capital'):
                        somme=somme+releve_compte_month.iloc[i]["Valeurs"]
                        
            #write df change years
            else:
                #write df        
                balance_df.loc[-1] = [year_i, month_i, somme]
                balance_df.index = balance_df.index + 1
                balance_df = balance_df.sort_index()
                #change years and month
                year_i=releve_compte_month.iloc[i]["year"]
                month_i=releve_compte_month.iloc[i]["month"]
                somme=0
                test=releve_compte_month.iloc[i]["Type de charge ou revenus"]
                if (test!='Epargne'and test!='Revenus du capital'):
                    somme=somme+releve_compte_month.iloc[i]["Valeurs"]

                    
        # write last row        
        balance_df.loc[-1] = [year_i, month_i, somme]
        balance_df.index = balance_df.index + 1
        balance_df = balance_df.sort_index(ascending=False) 
        balance_df = balance_df.reset_index()  
        return balance_df
    
    
#Not update    
# if __name__=='__main__':
#     fichier="notebooks/classifiction_todo_21.csv"
#     result_df=Account().categorize_new_bank_statement(fichier)
