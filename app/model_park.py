import os
import joblib
import numpy as np
import pandas as pd
import log_utils as lu

import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


nltk.download('wordnet')
nltk.download('stopwords')
en_stopwords = set(stopwords.words("english"))
wn_lemmatizer = WordNetLemmatizer()


class DataHandler:
    def __init__(self):
        self.train_df_path = os.path.join('..', 'data', 'train.csv')

    def load_train_dataset(self):
        df = pd.read_csv(self.train_df_path)
        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df['text'].apply(self._convert_emoticons)
        df['text'] = df['text'].apply(self._convert_emojis)
        df['clean_text'] = df['text'].apply(self._lemmatize_text)
        df['clean_text'] = df['clean_text'].apply(self._clean_text)
        df['text_len'] = df['clean_text'].apply(len)

        df['keyword'].fillna('UNKNOW', inplace=True)
        df['location'].fillna('UNKNOW', inplace=True)
        lu.logger.info(f"Prepared df shape: {df.shape}")
        return  df

    # Конвертация эмодзи в слова
    def _convert_emojis(self, text: str) -> str:
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
            return text
        
    # Конвертация эмоций со скобочками в словаs    
    def _convert_emoticons(self, text: str) -> str:
        for emot in EMOTICONS_EMO:
            text = text.replace(emot, EMOTICONS_EMO[emot].replace(" ","_"))
            return text

    def _clean_text(self, text: str) -> str:
        text = str(text).lower()
        clean_text = text.replace('{html}',"") # removing html files
        clean_text = re.sub(fr'[{string.punctuation}]', '', clean_text) # Removing punctuation
        clean_text = re.sub(r'http\S+', '',clean_text) # Removing links
        clean_text = re.sub('[0-9]+', '', clean_text)  # Removing numbers
        return clean_text

    def _lemmatize_text(self, text: str) -> str:
        return " ".join([wn_lemmatizer.lemmatize(w) for w in text.split() if w not in en_stopwords])


class ModelKeeper:
    def __init__(self, model_codename: str):
        self.model_codename = model_codename.lower()
        self.model_base_path = os.path.join('..', 'models')
        self.cat_feats = ['keyword', 'location']
        self.num_feats = ['text_len']
        self.text_feats = 'clean_text'
        self.target_col = 'target'
        self.model = None

    def build_model(self, data: pd.DataFrame, train=False, do_cv=False):
        os.makedirs(self.model_base_path, exist_ok=True)
        model_path = os.path.join(self.model_base_path, f"{self.model_codename}_pipeline.joblib")

        if not os.path.exists(model_path) or train:
            self.model = self._train_model(data, do_cv)
            lu.logger.info(f"[{self.model_codename}] Dumping to {model_path}...")
            joblib.dump(self.model, model_path, compress=1)
        else:
            lu.logger.info(f"[{self.model_codename}] Loading from {model_path}...")
            self.model = joblib.load(model_path)
        return self.model

    def _train_model(self, data: pd.DataFrame, do_cv: bool=True):
        lu.logger.info(f"[{self.model_codename}] Building pipeline...")
        model = self._build_pipeline()

        if do_cv:
            lu.logger.info(f"[{self.model_codename}] Cross_validate...")
            self._cross_validate_model(model, data)

        lu.logger.info(f"[{self.model_codename}] Fitting model ...")
        model.fit(data.drop(self.target_col, axis=1), data[self.target_col])

        lu.logger.info(f"[{self.model_codename}] Training finished.")
        return model

    def _build_pipeline(self) -> Pipeline:
        # CATEGORICAL
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('label_encoder', OneHotEncoder(handle_unknown='ignore')),
        ])
        # NUMERIC
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler())
        ])
        # TEXT 
        text_transformer = Pipeline(steps=[
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=20_000)),
            ('svd', TruncatedSVD(n_components=1000, random_state=42))
        ])
        # MODEL
        model = LogisticRegression(
            C=1.5,
            penalty='l1',
            class_weight='balanced',
            solver='liblinear',
            random_state=42,
        )
        # PIPELINE
        pipeline = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(transformers=[
                ('cat', categorical_transformer, self.cat_feats),
                ('num', numeric_transformer, self.num_feats),
                ('text', text_transformer, self.text_feats),
            ], remainder='drop', verbose=False)), 
            ('clf', model)
        ])
        return pipeline

    def _cross_validate_model(self, model, data: pd.DataFrame):
        scores = cross_validate(
            estimator=model, 
            X=data.drop(self.target_col, axis=1), 
            y=data[self.target_col],
            scoring=('accuracy', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'roc_auc'),
            cv=KFold(n_splits=8, shuffle=True, random_state=42),
            return_train_score=True
        )  # TODO return_estimator?

        scores_df = pd.DataFrame(scores)
        lu.logger.info(f"[{self.model_codename}]\n{scores_df}")
        lu.logger.info(f"[{self.model_codename}]\n{scores_df.describe()}")


if __name__ == '__main__':
    pass
    # lu.init_logging('ml_module', 'model_park.log')
    
    # # Train Model
    # data_handler = DataHandler()
    # train_df = data_handler.load_train_dataset()
    # prepared_df = data_handler.prepare_data(train_df)
    # model_keeper = ModelKeeper("Log_Reg_model")
    # model = model_keeper.build_model(prepared_df, train=True, do_cv=True)

    # # Use Model
    # data = {
    #     'keyword': ['aaa'], 
    #     'location': ['NewsYork'], 
    #     'text': ["Our Deeds are the Reason of this #earthquake M"]
    # }
    # df = pd.DataFrame(data)
    # data_handler = DataHandler()
    # prepared_df = data_handler.prepare_data(df)
    # model_keeper = ModelKeeper("Log_Reg_model")
    # model = model_keeper.build_model(prepared_df, train=False, do_cv=False)
    # print(model.predict_proba(df))