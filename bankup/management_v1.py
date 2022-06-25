#import
import os
#Pandas
import pandas as pd
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
        """
        self.categorisation_definition_df = None
        #self.X = X
        
    def categorisation_definition(self):
        """Import dataframe with the architecture of categorisation.
           
        """
        structure=[
                    "Revenus",
                    "Revenus du capital",
                    "Habitation",
                    "Assurance",
                    "Garde enfants",
                    "Scolarité",
                    "Impot",
                    "Alimentation",
                    "Entretien du logement",
                    "Amenagement du logement",
                    "Equipement du logement",
                    "Transport",
                    "Vehicule",
                    "Achat vehicule",
                    "Habillement",
                    "Beaute et bien etre",
                    "Sante",
                    "Culture, loisirs, education et sport",
                    "Moka",
                    "Vacances",
                    "Epargne cour terme",
                    "Epargne moyen terme",
                    "Epargne long terme",
                    "Divers",
                    "Service",
                    "Poste",
                    "Frais professionnel",
                    "Amende",
                    "Evenement de vie",
                ]
        type_Revenus=[
                    "Salaire arnaud",
                    "Salaire marine",
                    "Pole emploi arnaud",
                    "Pole emploi marine",
                    "Allocation Familial",
                    "Remise de cheque",
                    "Remise espece",
                    "Interessement, particiption",
                ]
        type_Revenus_du_capital=[
                    "Vente barcelone",
                    "Loyer pinel"
                ]
        type_Habitation=[
                        "Location avignon",
                        "Location toulouse",
                        "Remboursement de pret barcelone",
                        "Edf",
                        "Eau",
                        "Copro",
                        "Abonnement internet et telephonie",
                        "Abonnement tele"
                    ]
        type_Assurance=[
                        "Assurance voiture arnaud",
                        "Assurance voiture marine",
                        "Assurance protection juridique",
                        "Assurance logement barcelone",
                        "Assurance logement avignon",
                        "Assurance scolaire"
                    ]
        
        type_Garde_enfants=[
                        "Creche", 
                        "Nounou",
                        "Babysister"
                    ]
        type_Scolarite=[
                        "Ecole arthur"
                    ]
        type_Impot=[
                    "Impot",
                ]
        type_Alimentation=[
                    "Alimentation domicile",
                    "Cantine",
                    "Cantine arthur",
                    "Resto",
                    "Mc do",
                    "Pizza",
                    "Glace",
                    "Boucherie",
                    "Boulangerie, patisserie, gateau et chocolat",
                    "Fromage",
                    "Caviste",
                    "Cafe et the"
                ]
        type_Entretien_du_logement=[
                            "Produit entretien logement",
                            "Bricolage, quincaillerie"
                        ]
        type_Amenagement_du_logement=[
                            "Plante, fleurs",
                            "Decoration",
                            "Couverture, drap",
                            "Affaire de bureau"
                        ]
        type_Equipement_du_logement=[
                        "Mobilier",
                        "Electromenager",
                        "Cuisine, vaisselle",
                        "Chambre arthur",
                        "Telephonie, PC et hardware",
                    ]
        type_Transport=[
                    "Essence",
                    "Transports en commun",
                    "Autoroute",
                    "Parking",
                ]
        type_Vehicule=[
                    "Entretien utilisation voiture"
                    ]
        type_Achat_vehicule=[
                    "Achat vehicule"
                ]
        type_Habillement=[
                        "Vetement arnaud",
                        "Vetement marine",
                        "Vetement arthur",
                        "Lingerie",
                        "Bijoux, montres, sac",
                        "Nettoyage, reparation et pressing vetement",
                    ]
        type_Beaute_et_bien_etre=[
                    "Hygiene, esthetic, cosmetique et massage",
                    "Coiffeur"
                    ]
        type_Sante=[
                    "Consultation de praticiens generaliste",
                    "Ophtalmologue et lunette",
                    "Gynecologie et accouchement",
                    "acupuncture",
                    "Neurologie",
                    "Podologue",
                    "Gastroenterologue",
                    "Pediatre",
                    "Medecin du sport",
                    "Cardiologue",
                    "Dentiste",
                    "Pharmacie",
                    "Radiographie",
                    "Laboratoire et clinique",
                    "Cpam et Mutelle",
                ]
        type_Culture_loisirs_education_et_sport=[
                    "Photo et album",
                    "Livres, disques, films, Jeux",
                    "Abonnement sport, licence et piscine",
                    "Abonnement sport, licence et piscine Arthur",
                    "Course sport",
                    "Vetement et equipement sport arnaud",
                    "Vetement et equipement sport marine",
                    "Journaux, revues",
                    "Tissus, couture et creation",
                    "Jeux, livres arthur",
                    "Sortie soiree, cinema et we",
                    "Musee, expo et culture",
                    "Formation, education, MOOC",
                    "CE arnaud",
                    "CE marine",
                    "Spectacle, concert, theatre",
                    "Parc attraction"
                ]
        type_Moka=[
                "Veterinaire",
                "Alimentation moka",
            ]
        type_Vacances=[
                    "Vacances, week-end",
                    "Voyage barcelone",
                    "Voyage New York",
                    "Voyage Tanzani",
                    "Hotel",
                    "Transport longue distance train",
                    "Transport longue distance avion",
                ]
        type_Epargne_cour_terme=[
                    "Livret A arnaud",
                    "Livret A marine",
                    "Livret DDS arnaud",
                    "Livret DDS marine",
                ]
        type_Epargne_moyen_terme=[
                    "Epargne entreprise arnaud",
                    "Epargne entreprise marine",
                    "Assurance vie arnaud",
                    "Assurance vie marine",
                ]
        type_Epargne_long_terme=[
                    "Remboursement de pret pinel",
                    "Complement pinel",
                    "Assurance logement pinel"
                ]
        type_Divers=[
                    "Cadeau",
                    "Retrait espece",
                    "Cheque non identifie",
                    "Virement aviles liberto elisabeth",
                    "Virement emile guillaume",
                    "Virement changement de compte",
                    "Coordonnier,cles,reparation",
                    "Dechetterie"
                ]
        type_Service=[
                    "Frais banque",
                    "Impression imprimante",
                    "Service google"
                ]
        type_Poste=[
                    "Poste"
                ]
        type_Frais_professionnel=[
                    "Deplacement pro",
                    "Vetement professionnel",
                    "Evenementiel"
                ]
        type_Amende=[
                    "Amende"
                ]
        type_Evenement_de_vie=[
                    "Mariage",
                    "Demenagement",
                ]
        
        dico={}
        dico[structure[0]]=type_Revenus
        dico[structure[1]]=type_Revenus_du_capital
        dico[structure[2]]=type_Habitation
        dico[structure[3]]=type_Assurance
        dico[structure[4]]=type_Garde_enfants
        dico[structure[5]]=type_Scolarite
        dico[structure[6]]=type_Impot
        dico[structure[7]]=type_Alimentation
        dico[structure[8]]=type_Entretien_du_logement
        dico[structure[9]]=type_Amenagement_du_logement
        dico[structure[10]]=type_Equipement_du_logement
        dico[structure[11]]=type_Transport
        dico[structure[12]]=type_Vehicule
        dico[structure[13]]=type_Achat_vehicule
        dico[structure[14]]=type_Habillement
        dico[structure[15]]=type_Beaute_et_bien_etre
        dico[structure[16]]=type_Sante
        dico[structure[17]]=type_Culture_loisirs_education_et_sport
        dico[structure[18]]=type_Moka
        dico[structure[19]]=type_Vacances
        dico[structure[20]]=type_Epargne_cour_terme
        dico[structure[21]]=type_Epargne_moyen_terme
        dico[structure[22]]=type_Epargne_long_terme
        dico[structure[23]]=type_Divers
        dico[structure[24]]=type_Service
        dico[structure[25]]=type_Poste
        dico[structure[26]]=type_Frais_professionnel
        dico[structure[27]]=type_Amende
        dico[structure[28]]=type_Evenement_de_vie

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
    
    
#Not update    
# if __name__=='__main__':
#     fichier="notebooks/classifiction_todo_21.csv"
#     result_df=Account().categorize_new_bank_statement(fichier)
