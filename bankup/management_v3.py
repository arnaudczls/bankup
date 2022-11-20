#import
import os
#Pandas
import pandas as pd
#Numpy
import numpy as np
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
        
    def balance_account(df,Year,Month,Day,Day_include=True):
        """
            Fonction qui permet d'avoir l'argent restant sur le compte en fonction du jour.
            Input:
                   df= dataframe avec les releve de compte
                   Year= l'année (par exemple 2022)
                   Month= le mois (par exemple 04)
                   Day= le jour  (par exemple 01)
                   Day_include= prendre en compte le jour compris
                   (Par defaut= Yes==> day=01, la fonction va perndre le jours 01 aussi dans le calcul (le 02 non))
        """
        # #import data
        # df=data.get_data(fichier,onglet)
        # df['Date']=pd.to_datetime(df['Date'],dayfirst=True)

        separation="-"
        #Date initial
        year_ini="2017"
        Month_ini="01"
        day_ini="07"

        valeur_ini=6146.96#value comes from RELEVES_0085792980_20170209.pdf

        date_ini=''.join([year_ini,separation,Month_ini,separation,day_ini])
        date_ini = pd.to_datetime(date_ini)

        #Date objectif
        date=''.join([str(Year),separation,str(Month),separation,str(Day)])
        date = pd.to_datetime(date)

        #filter
        if Day_include==True:
            date_filter = np.logical_and(pd.to_datetime(df['Date'],dayfirst=True) > date_ini,
                                         pd.to_datetime(df['Date'],dayfirst=True) <= date)
        else:
            date_filter = np.logical_and(pd.to_datetime(df['Date'],dayfirst=True) > date_ini,
                                         pd.to_datetime(df['Date'],dayfirst=True) < date)
        
        df = df[date_filter]
        df=df.reset_index(drop=True)

        #Somme
        Somme=valeur_ini+df["Valeurs"].sum()
        
        return Somme    
        
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
            "Scolarite":"Charges fixes",
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

    def get_analyze_cost_income_by_month (releve_compte_month):
        #----------info---------------
        #Calculate balance of statement by month.
        #Calculation of each cost and income to see at the end of the month, there are more cost than income.
        # 
        #The summe of income and cost is done each month   
        #    
        #Warning!!!!
        #the caterogry Epargne and Revenus du capital are exclut for the calculation
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
        liste_columns=["year","month","benefice lost","Revenus","Sommes charge",
               "Charges courantes","Alimentation","Beaute et bien etre","Culture, loisirs, education et sport","Habillement","Sante","Transport","Entretien du logement","Moka",
               "Charges fixes","Assurance","Epargne long terme","Garde enfants","Habitation","Impot","Service","Scolarite",
               "Charges occasionnelles","Amenagement du logement","Divers","Frais professionnel","Poste","Vehicule",
               "Charges expectionnelles","Amende","Vacances","Equipement du logement","Achat vehicule","Evenement de vie"
              ]

        balance_df=pd.DataFrame(columns=liste_columns)
        year_i=releve_compte_month.iloc[0]["year"]
        month_i=releve_compte_month.iloc[0]["month"]
        #Benefice perte
        somme_benefice_perte=0
        #revenu
        revenu=0
        #Somme charge
        somme_charge=0
        #Charges_courante
        somme_charge_courant=0
        somme_alimentation=0
        somme_entretien_logement=0
        somme_sante=0
        somme_transport=0
        somme_culture_sport=0
        somme_beaute=0
        somme_moka=0
        somme_habillement=0
        #Charges_fixes
        somme_charges_fixes=0
        somme_habitation=0
        somme_assurance=0
        somme_garde_enfants=0
        somme_scolarite=0
        somme_impot=0
        somme_epargne_long=0
        somme_service=0
        #Charges_occasionnelles
        somme_charges_occasionnelles=0
        somme_amenagement_logement=0
        somme_vehicule=0
        somme_divers=0
        somme_poste=0
        somme_frais_professionnel=0
        #charge_expectionnelles
        somme_charges_expectionnelles=0
        somme_equipement_logement=0
        somme_evenement_vie=0
        somme_vacances=0
        somme_amende=0
        somme_achat_vehicule=0


        #loop for
        for i in range(len(releve_compte_month)):
            #years
            if releve_compte_month.iloc[i]["year"]==year_i:
                #month
                if releve_compte_month.iloc[i]["month"]==month_i:
                    test=releve_compte_month.iloc[i]["Type de charge ou revenus"]
                    test_structure=releve_compte_month.iloc[i]["Groupes par structure"]
                    #Benefice perte
                    if (test!='Epargne'and test!='Revenus du capital'):
                        somme_benefice_perte=somme_benefice_perte+releve_compte_month.iloc[i]["Valeurs"]
                    #Somme charge
                    if (test=='Charges courantes'or test=='Charges fixes'or test=="Charges occasionnelles" or test=="Charges expectionnelles"):
                        somme_charge=somme_charge+releve_compte_month.iloc[i]["Valeurs"]
                    #revenus
                    if (test=='Revenus'):
                        revenu=revenu+releve_compte_month.iloc[i]["Valeurs"]    
                    #charge courant   
                    if test=="Charges courantes":
                        somme_charge_courant=somme_charge_courant+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Alimentation":
                            somme_alimentation=somme_alimentation+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Beaute et bien etre":
                            somme_beaute=somme_beaute+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Culture, loisirs, education et sport":
                            somme_culture_sport=somme_culture_sport+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Habillement":
                            somme_habillement=somme_habillement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Sante":
                            somme_sante=somme_sante+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Transport":
                            somme_transport=somme_transport+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Entretien du logement":
                            somme_entretien_logement=somme_entretien_logement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Moka":
                            somme_moka=somme_moka+releve_compte_month.iloc[i]["Valeurs"]
                    #charge fixes   
                    if test=="Charges fixes":
                        somme_charges_fixes=somme_charges_fixes+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Assurance":
                            somme_assurance=somme_assurance+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Epargne long terme":
                            somme_epargne_long=somme_epargne_long+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Garde enfants":
                            somme_garde_enfants=somme_garde_enfants+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Habitation":
                            somme_habitation=somme_habitation+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Impot":
                            somme_impot=somme_impot+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Service":
                            somme_service=somme_service+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Scolarite":
                            somme_scolarite=somme_scolarite+releve_compte_month.iloc[i]["Valeurs"]
                    #charge occasionnelles   
                    if test=="Charges occasionnelles":
                        somme_charges_occasionnelles=somme_charges_occasionnelles+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Amenagement du logement":
                            somme_amenagement_logement=somme_amenagement_logement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Divers":
                            somme_divers=somme_divers+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Frais professionnel":
                            somme_frais_professionnel=somme_frais_professionnel+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Poste":
                            somme_poste=somme_poste+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Vehicule":
                            somme_vehicule=somme_vehicule+releve_compte_month.iloc[i]["Valeurs"]
                    #charge expectionnelles   
                    if test=="Charges expectionnelles":
                        somme_charges_expectionnelles=somme_charges_expectionnelles+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Amende":
                            somme_amende=somme_amende+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Vacances":
                            somme_vacances=somme_vacances+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Equipement du logement":
                            somme_equipement_logement=somme_equipement_logement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Achat vehicule":
                            somme_achat_vehicule=somme_achat_vehicule+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Evenement de vie":
                            somme_evenement_vie=somme_evenement_vie+releve_compte_month.iloc[i]["Valeurs"]
                                            
                            
                # write df and change month
                else:
                    #write df        
                    balance_df.loc[-1] =[year_i, month_i, somme_benefice_perte,revenu,somme_charge,
                                        somme_charge_courant,somme_alimentation,somme_beaute,somme_culture_sport,somme_habillement,somme_sante,somme_transport,somme_entretien_logement,somme_moka,
                                        somme_charges_fixes,somme_assurance,somme_epargne_long,somme_garde_enfants,somme_habitation,somme_impot,somme_service,somme_scolarite,
                                        somme_charges_occasionnelles,somme_amenagement_logement,somme_divers,somme_frais_professionnel,somme_poste,somme_vehicule,
                                        somme_charges_expectionnelles,somme_amende,somme_vacances,somme_equipement_logement,somme_achat_vehicule,somme_evenement_vie
                                        ]
                    balance_df.index = balance_df.index + 1
                    balance_df = balance_df.sort_index()
                    #change month
                    month_i=releve_compte_month.iloc[i]["month"]
                    #Benefice perte
                    somme_benefice_perte=0
                    #revenu
                    revenu=0
                    #Somme charge
                    somme_charge=0
                    #Charges_courante
                    somme_charge_courant=0
                    somme_alimentation=0
                    somme_entretien_logement=0
                    somme_sante=0
                    somme_transport=0
                    somme_culture_sport=0
                    somme_beaute=0
                    somme_moka=0
                    somme_habillement=0
                    #Charges_fixes
                    somme_charges_fixes=0
                    somme_habitation=0
                    somme_assurance=0
                    somme_garde_enfants=0
                    somme_scolarite=0
                    somme_impot=0
                    somme_epargne_long=0
                    somme_service=0
                    #Charges_occasionnelles
                    somme_charges_occasionnelles=0
                    somme_amenagement_logement=0
                    somme_vehicule=0
                    somme_divers=0
                    somme_poste=0
                    somme_frais_professionnel=0
                    #charge_expectionnelles
                    somme_charges_expectionnelles=0
                    somme_equipement_logement=0
                    somme_evenement_vie=0
                    somme_vacances=0
                    somme_amende=0
                    somme_achat_vehicule=0
                    
                    test=releve_compte_month.iloc[i]["Type de charge ou revenus"]
                    test_structure=releve_compte_month.iloc[i]["Groupes par structure"]
                    #Benefice perte
                    if (test!='Epargne'and test!='Revenus du capital'):
                        somme_benefice_perte=somme_benefice_perte+releve_compte_month.iloc[i]["Valeurs"]
                    #Somme charge
                    if (test=='Charges courantes'or test=='Charges fixes'or test=="Charges occasionnelles" or test=="Charges expectionnelles"):
                        somme_charge=somme_charge+releve_compte_month.iloc[i]["Valeurs"]
                    #revenus
                    if (test=='Revenus'):
                        revenu=revenu+releve_compte_month.iloc[i]["Valeurs"]
                    #charge courant   
                    if test=="Charges courantes":
                        somme_charge_courant=somme_charge_courant+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Alimentation":
                            somme_alimentation=somme_alimentation+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Beaute et bien etre":
                            somme_beaute=somme_beaute+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Culture, loisirs, education et sport":
                            somme_culture_sport=somme_culture_sport+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Habillement":
                            somme_habillement=somme_habillement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Sante":
                            somme_sante=somme_sante+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Transport":
                            somme_transport=somme_transport+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Entretien du logement":
                            somme_entretien_logement=somme_entretien_logement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Moka":
                            somme_moka=somme_moka+releve_compte_month.iloc[i]["Valeurs"]
                    #charge fixes   
                    if test=="Charges fixes":
                        somme_charges_fixes=somme_charges_fixes+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Assurance":
                            somme_assurance=somme_assurance+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Epargne long terme":
                            somme_epargne_long=somme_epargne_long+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Garde enfants":
                            somme_garde_enfants=somme_garde_enfants+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Habitation":
                            somme_habitation=somme_habitation+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Impot":
                            somme_impot=somme_impot+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Service":
                            somme_service=somme_service+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Scolarite":
                            somme_scolarite=somme_scolarite+releve_compte_month.iloc[i]["Valeurs"]
                    #charge occasionnelles   
                    if test=="Charges occasionnelles":
                        somme_charges_occasionnelles=somme_charges_occasionnelles+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Amenagement du logement":
                            somme_amenagement_logement=somme_amenagement_logement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Divers":
                            somme_divers=somme_divers+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Frais professionnel":
                            somme_frais_professionnel=somme_frais_professionnel+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Poste":
                            somme_poste=somme_poste+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Vehicule":
                            somme_vehicule=somme_vehicule+releve_compte_month.iloc[i]["Valeurs"]
                    #charge expectionnelles   
                    if test=="Charges expectionnelles":
                        somme_charges_expectionnelles=somme_charges_expectionnelles+releve_compte_month.iloc[i]["Valeurs"] 
                        if test_structure=="Amende":
                            somme_amende=somme_amende+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Vacances":
                            somme_vacances=somme_vacances+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Equipement du logement":
                            somme_equipement_logement=somme_equipement_logement+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Achat vehicule":
                            somme_achat_vehicule=somme_achat_vehicule+releve_compte_month.iloc[i]["Valeurs"]
                        if test_structure=="Evenement de vie":
                            somme_evenement_vie=somme_evenement_vie+releve_compte_month.iloc[i]["Valeurs"]
                        
            #write df change years
            else:
                #write df        
                balance_df.loc[-1] =[year_i, month_i, somme_benefice_perte,revenu,somme_charge,
                                    somme_charge_courant,somme_alimentation,somme_beaute,somme_culture_sport,somme_habillement,somme_sante,somme_transport,somme_entretien_logement,somme_moka,
                                    somme_charges_fixes,somme_assurance,somme_epargne_long,somme_garde_enfants,somme_habitation,somme_impot,somme_service,somme_scolarite,
                                    somme_charges_occasionnelles,somme_amenagement_logement,somme_divers,somme_frais_professionnel,somme_poste,somme_vehicule,
                                    somme_charges_expectionnelles,somme_amende,somme_vacances,somme_equipement_logement,somme_achat_vehicule,somme_evenement_vie
                                    ]
                balance_df.index = balance_df.index + 1
                balance_df = balance_df.sort_index()
                #change years and month
                year_i=releve_compte_month.iloc[i]["year"]
                month_i=releve_compte_month.iloc[i]["month"]
                #Benefice perte
                somme_benefice_perte=0
                #revenu
                revenu=0
                #Somme charge
                somme_charge=0
                #Charges_courante
                somme_charge_courant=0
                somme_alimentation=0
                somme_entretien_logement=0
                somme_sante=0
                somme_transport=0
                somme_culture_sport=0
                somme_beaute=0
                somme_moka=0
                somme_habillement=0
                #Charges_fixes
                somme_charges_fixes=0
                somme_habitation=0
                somme_assurance=0
                somme_garde_enfants=0
                somme_scolarite=0
                somme_impot=0
                somme_epargne_long=0
                somme_service=0
                #Charges_occasionnelles
                somme_charges_occasionnelles=0
                somme_amenagement_logement=0
                somme_vehicule=0
                somme_divers=0
                somme_poste=0
                somme_frais_professionnel=0
                #charge_expectionnelles
                somme_charges_expectionnelles=0
                somme_equipement_logement=0
                somme_evenement_vie=0
                somme_vacances=0
                somme_amende=0
                somme_achat_vehicule=0
                
                test=releve_compte_month.iloc[i]["Type de charge ou revenus"]
                test_structure=releve_compte_month.iloc[i]["Groupes par structure"]
                #Benefice perte
                if (test!='Epargne'and test!='Revenus du capital'):
                    somme_benefice_perte=somme_benefice_perte+releve_compte_month.iloc[i]["Valeurs"]
                #Somme charge
                    if (test=='Charges courantes' or test=='Charges fixes'or test=="Charges occasionnelles" or test=="Charges expectionnelles"):
                        somme_charge=somme_charge+releve_compte_month.iloc[i]["Valeurs"]
                #revenue
                if (test=='Revenus'):
                    revenu=revenu+releve_compte_month.iloc[i]["Valeurs"]
                #charge courant   
                if test=="Charges courantes":
                    somme_charge_courant=somme_charge_courant+releve_compte_month.iloc[i]["Valeurs"] 
                    if test_structure=="Alimentation":
                        somme_alimentation=somme_alimentation+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Beaute et bien etre":
                        somme_beaute=somme_beaute+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Culture, loisirs, education et sport":
                        somme_culture_sport=somme_culture_sport+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Habillement":
                        somme_habillement=somme_habillement+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Sante":
                        somme_sante=somme_sante+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Transport":
                        somme_transport=somme_transport+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Entretien du logement":
                        somme_entretien_logement=somme_entretien_logement+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Moka":
                        somme_moka=somme_moka+releve_compte_month.iloc[i]["Valeurs"]
                #charge fixes   
                if test=="Charges fixes":
                    somme_charges_fixes=somme_charges_fixes+releve_compte_month.iloc[i]["Valeurs"] 
                    if test_structure=="Assurance":
                        somme_assurance=somme_assurance+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Epargne long terme":
                        somme_epargne_long=somme_epargne_long+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Garde enfants":
                        somme_garde_enfants=somme_garde_enfants+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Habitation":
                        somme_habitation=somme_habitation+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Impot":
                        somme_impot=somme_impot+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Service":
                        somme_service=somme_service+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Scolarite":
                        somme_scolarite=somme_scolarite+releve_compte_month.iloc[i]["Valeurs"]
                #charge occasionnelles   
                if test=="Charges occasionnelles":
                    somme_charges_occasionnelles=somme_charges_occasionnelles+releve_compte_month.iloc[i]["Valeurs"] 
                    if test_structure=="Amenagement du logement":
                        somme_amenagement_logement=somme_amenagement_logement+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Divers":
                        somme_divers=somme_divers+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Frais professionnel":
                        somme_frais_professionnel=somme_frais_professionnel+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Poste":
                        somme_poste=somme_poste+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Vehicule":
                        somme_vehicule=somme_vehicule+releve_compte_month.iloc[i]["Valeurs"]
                #charge expectionnelles   
                if test=="Charges expectionnelles":
                    somme_charges_expectionnelles=somme_charges_expectionnelles+releve_compte_month.iloc[i]["Valeurs"] 
                    if test_structure=="Amende":
                        somme_amende=somme_amende+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Vacances":
                        somme_vacances=somme_vacances+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Equipement du logement":
                        somme_equipement_logement=somme_equipement_logement+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Achat vehicule":
                        somme_achat_vehicule=somme_achat_vehicule+releve_compte_month.iloc[i]["Valeurs"]
                    if test_structure=="Evenement de vie":
                        somme_evenement_vie=somme_evenement_vie+releve_compte_month.iloc[i]["Valeurs"]
                    
        # write last row        
        balance_df.loc[-1] = [year_i, month_i, somme_benefice_perte,revenu,somme_charge,
                            somme_charge_courant,somme_alimentation,somme_beaute,somme_culture_sport,somme_habillement,somme_sante,somme_transport,somme_entretien_logement,somme_moka,
                            somme_charges_fixes,somme_assurance,somme_epargne_long,somme_garde_enfants,somme_habitation,somme_impot,somme_service,somme_scolarite,
                            somme_charges_occasionnelles,somme_amenagement_logement,somme_divers,somme_frais_professionnel,somme_poste,somme_vehicule,
                            somme_charges_expectionnelles,somme_amende,somme_vacances,somme_equipement_logement,somme_achat_vehicule,somme_evenement_vie
                            ]
        balance_df.index = balance_df.index + 1
        balance_df = balance_df.sort_index(ascending=False)
        return balance_df
    
    
#Not update    
# if __name__=='__main__':
#     fichier="notebooks/classifiction_todo_21.csv"
#     result_df=Account().categorize_new_bank_statement(fichier)
