from sklearn.base import TransformerMixin, BaseEstimator
import unidecode
import string
from nltk.stem import WordNetLemmatizer

class Pipeline_clean_text(TransformerMixin, BaseEstimator):
# TransformerMixin generates a fit_transform method from fit and transform
# BaseEstimator generates get_params and set_params methods
        #----------info---------------
        #creer uune class pipeline pour nettoyer les donnes textuel
        #   Nettoyage:
        #       1) minuscule
        #       2) Enleve la poncutation
        #       3) Enleve les accents
        #       4) Enleve les mot de la stop liste que l'on fournie
        #           (en parametre)
        #
        #
        #Mise a jour: 05/07/2021
        #----------end info---------------
    def __init__(self, list_stop_word=None):
        self.list_stop_word = list_stop_word


    def fit(self, X, y=None):
    
        return self

    def transform(self, X, y=None):

        def function_clean_text(x,list_stop_word=None):
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
            if list_stop_word:
                stop_words = set(list_stop_word) 
                word_tokens = x.split()
                x = " ".join([w for w in word_tokens if not w in stop_words])
            
            #     #Garde la racine
            #     lemmatizer = WordNetLemmatizer()
            #     lemmatized = [lemmatizer.lemmatize(word) for word in x]
            #     x = "".join(lemmatized)
            return x

        X.apply(function_clean_text,self.list_stop_word)
        return X

    
# if __name__=='__main__':
#      fichier='releves_banque_2015_16_17_18_19_20_21_V6.xls'
#      onglet='Releve_compte'
#      data=Classification_model.classification_by_enseigne(fichier,onglet)
# #     data=Account_statement.get_data_clean(fichier,onglet)
# #     print(data)