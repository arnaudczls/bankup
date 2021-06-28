import os
import pandas as pd
import numpy as np

class Account_statement:
    def __init__(self):
        pass

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
        path_v2=".."
        path_v3 = "raw_data"
        # Join various path components 
        fichier=os.path.join(path_v2,path_v3, fichier)
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
    
    
if __name__=='__main__':
    fichier='releves_banque_2015_16_17_18_19_20_21_V6.xlsm'
    onglet='Releve_compte'
    data=Account_statement.get_data(fichier,onglet)
    print(data)