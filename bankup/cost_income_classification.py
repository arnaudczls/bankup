import os
import pandas as pd
import numpy as np
import re
from sklearn.base import TransformerMixin, BaseEstimator
import unidecode
import string
from nltk.stem import WordNetLemmatizer

class Pipeline_clean_text(TransformerMixin, BaseEstimator):
# TransformerMixin generates a fit_transform method from fit and transform
# BaseEstimator generates get_params and set_params methods
    def __init__(self, list_stop_word):
        self.list_stop_word = list_stop_word


    def fit(self, X, y=None):
    
    return self

    def transform(self, X, y=None):
        #lower==> minuscule
        x=x.lower()
        
        #punctuation
        for punctuation in string.punctuation:
            x = x.replace(punctuation, ' ')  
                
        # remove accents    
        unaccented_string = unidecode.unidecode(x)
        
        #enleve les digits
        #methode 1
        #x=re.sub(r'\d+', '', x)
        #methode 2
        x=''.join(word for word in x if not word.isdigit())
        
        #enleve les stop words
        stop_words = set(list_stop_word) 
        word_tokens = x.split()
        x = " ".join([w for w in word_tokens if not w in stop_words])
        
    #     #Garde la racine
    #     lemmatizer = WordNetLemmatizer()
    #     lemmatized = [lemmatizer.lemmatize(word) for word in x]
    #     x = "".join(lemmatized)
        return x


    def classification_by_enseigne(fichier,onglet):
        #----------info---------------
        #fait un model d'analyse textuel pour classificer tous les label pour avoir les enseignes
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
        releve_compte=Account_statement.get_data_clean(fichier,onglet)
        #traitement des label
        #Supression des nombres
        #Suppresssion des nombres pour le groupe label
        releve_compte_clean=releve_compte_clean.Label.
                apply(lambda x: re.sub(r'\d+', '', x))

        
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
    
    def get_data_clean(fichier,onglet):
        #----------info---------------
        #Recup les data du fichier excel et nettoie la base de donnée
        #   -les dates au bon format
        #   -efface les lignes qui son identifier:
        #                       Autre
        #                       Cheque non identifie
        #Input:
        #fichier==>Nom du fichier
        #onglet==>Onglet du fichier
        #
        #Output:
        #Dataframe
        #
        #Mise a jour: 29/06/2021
        #----------end info---------------
        releve_compte=Account_statement.get_data(fichier,onglet)
        # Traitement des dates
        releve_compte['Date']=pd.to_datetime(releve_compte['Date'])
        #releve_compte['Date']=pd.to_datetime(releve_compte['Date'],format='%Y-%m-%d')
        #Supprime les lignes non utilisé
        #Suppression des lignes autre et cheque non identifier
        Mask_liste_a_suprimer=(
                            releve_compte["Groupes par type"].str.contains(
                            "Autre|Cheque non identifie"
                                    )
                                 )
        releve_compte=releve_compte[Mask_liste_a_suprimer == False]
        return releve_compte
    
    
if __name__=='__main__':
     fichier='releves_banque_2015_16_17_18_19_20_21_V6.xls'
     onglet='Releve_compte'
     data=Classification_model.classification_by_enseigne(fichier,onglet)
#     data=Account_statement.get_data_clean(fichier,onglet)
#     print(data)