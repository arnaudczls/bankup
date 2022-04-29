from ipaddress import collapse_addresses
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
#Traitement de texte
import unidecode
import string
from nltk.stem import WordNetLemmatizer
#package
# import bankup.data as data


def get_data(fichier,onglet):
    #----------info---------------
    #Recup les data du fichier excel et supprimes les lignes et colonnes nul
    #
    #Input:
    #fichier==>Nom du fichier
    #onglet==>Onglet du fichier
    #
    #Output:
    #Dataframe
    #
    #Mise a jour: 29/06/2021
    #----------end info---------------
    # Path
    path=Path(__file__).parents[1]
    path_v2 = "raw_data"
    # Join various path components 
    fichier=os.path.join(path,path_v2, fichier)
    Excel_releve_compte = pd.ExcelFile(fichier)
    releve_compte=pd.read_excel(Excel_releve_compte, onglet)
    #Prend la ligne que l'on va utilisé pour la metre en entete
    columns=list(releve_compte.iloc[1])
    
    #Supprime les lignes et les columns vides
    raw_to_drop=[0,1,2,3, 4]
    columns_to_keep=[1,2,3,4,5,6,10]
    columns=list(releve_compte.iloc[1,columns_to_keep])
    releve_compte=releve_compte.drop(index=raw_to_drop)
    releve_compte=releve_compte.iloc[:,columns_to_keep]
    #Supprime les lignes ou il n'y a riens
    releve_compte=releve_compte.dropna(how = 'all')
    #reset de l'index
    releve_compte=releve_compte.reset_index(drop=True)
    
    #Renome la ligne entete
    releve_compte.columns=columns
    return releve_compte
    
def get_data_clean(fichier,onglet,drop_row_null="yes"):
    #----------info---------------
    #Recup les data du fichier excel et nettoie la base de donnée
    #   -les dates au bon format
    #       Rajoute une colonnes years, month, day
    #   -efface les lignes qui son identifier: argument drop_row_null
    #                       Autre
    #                       Cheque non identifie
    #Input:
    #fichier==>Nom du fichier
    #onglet==>Onglet du fichier
    #drop_row_null==> efface les lignes contenant "Autre ou Cheque non identifie" de la colonne"Groupes par type"
    #                 par default "yes"
    #                 possible choise: "yes" or "no"
    #
    #Output:
    #Dataframe
    #
    #Mise a jour: 29/06/2021
    #----------end info---------------
    releve_compte=get_data(fichier,onglet)
    # Traitement des dates
    releve_compte['Date']=pd.to_datetime(releve_compte['Date'])
    #releve_compte['Date']=pd.to_datetime(releve_compte['Date'],format='%Y-%m-%d')
    #Rajoute une colonnes years, month and day
    releve_compte["year"]=releve_compte["Date"].dt.year
    releve_compte["month"]=releve_compte["Date"].dt.month
    releve_compte["day"]=releve_compte["Date"].dt.day
    
    # Traitement des valeurs
    releve_compte['Valeurs']=pd.to_numeric(releve_compte['Valeurs'],downcast="float")
    #Supprime les lignes non utilisé
    #Suppression des lignes autre et cheque non identifier
    if drop_row_null=="yes":
        Mask_liste_a_suprimer=(
                            releve_compte["Groupes par type"].str.contains(
                            "Autre|Cheque non identifie"
                                    )
                                )
        releve_compte=releve_compte[Mask_liste_a_suprimer == False]
    return releve_compte


def get_data_by_month(fichier,onglet):
    #----------info---------------
    #Renvoie un Dataframe avec les depenses et revenus par mois
    # les colonnes Date, Label,Groupes par enseigne et Groupes par type sont effacées
    #
    #
    #
    #Input:
    #fichier==>Nom du fichier
    #onglet==>Onglet du fichier
    #
    #Output:
    #Dataframe
    #
    #Mise a jour: 29/06/2021
    #----------end info---------------
    releve_compte=get_data_clean(fichier,onglet)
    #Group by
    releve_compte=releve_compte.groupby(
        by=["year","month","Type de charge ou revenus","Groupes par structure"]).sum()
    releve_compte=releve_compte.drop(columns=["day"])
    releve_compte=releve_compte.reset_index()
    return releve_compte

def get_data_cost_by_month(fichier,onglet):
    #----------info---------------
    #Renvoie un Dataframe avec les depenses
    #les revenues sont effaces
    #les depenses sont transformer en >0
    #Enleve les evenements exeptionel (achat véhicule,Mariage etc.) et valeurs extremes
    #       
    #
    #
    #Input:
    #fichier==>Nom du fichier
    #onglet==>Onglet du fichier
    #
    #Output:
    #Dataframe
    #
    #Mise a jour: 29/06/2021
    #----------end info---------------
    releve_compte=get_data_by_month(fichier,onglet)
    #les revenues sont effaces
    releve_compte=releve_compte[releve_compte['Valeurs']<0]
    #les depenses sont transformer en >0
    releve_compte['Valeurs']=abs(releve_compte['Valeurs'])
    releve_compte=releve_compte.reset_index(drop=True)
    #Enleve les evenements exeptionel
    releve_compte=releve_compte[releve_compte["Groupes par structure"]\
    .str.contains("Revenus du capital|Achat vehicule|Evenement de vie")== False]
    releve_compte=releve_compte.reset_index(drop=True)
    #Enleve les valeurs extremes
    #enleve les valeur extreme (virement)
    releve_compte=releve_compte[releve_compte.Valeurs.between(0, 9_000)]
    releve_compte=releve_compte.reset_index(drop=True)

    return releve_compte


def clean_data_text(X):
    #----------info---------------
    #Renvoie un Serie en:
    #   En minuscule
    #   Sans digidit
    #   Sans pontuation
    #   Sans accent
    #       
    #Utilisé pour nettoyer la colonne Label
    #
    #Input:
    #Serie==>Nom du fichier
    #
    #Output:
    #Serie ==> Propre
    #
    #Mise a jour: 29/06/2021
    #----------end info---------------
            #lower==> minuscule
            X=X.str.lower()
            
            #punctuation
            def punctuation(x):
                for punctuation in string.punctuation:
                    x = x.replace(punctuation, ' ')
                return x  
            X=X.apply(punctuation)
                   
            # remove accents    
            X=X.apply(unidecode.unidecode)
            
            #enleve les digits   
            #methode 1
            #X=X.apply(lambda x: re.sub(r'\d+', '', x))
            #methode 2
            X=X.apply(lambda x:
                ''.join([word for word in x if not word.isdigit()])
                )
            
            #     #Garde la racine
            #     lemmatizer = WordNetLemmatizer()
            #     lemmatized = [lemmatizer.lemmatize(word) for word in x]
            #     x = "".join(lemmatized)
            return X
   
    
if __name__=='__main__':
     fichier='releves_banque_2015_16_17_18_19_20_21_V7.xls'
     onglet='Releve_compte'
     #data=data.get_data(fichier,onglet)
     data=get_data_clean(fichier,onglet)
     print(data)
     data.Label=clean_data_text(data.Label)
     print(data)