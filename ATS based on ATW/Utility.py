import os
import numpy as np  # Fundamental package for scientific computing with Python
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import calendar
import time

def dumpModel(name, model):
    joblib.dump(model, name)

def loadModel(name):
    return joblib.load(name)

def convert_csv_to_xlsx(csv_file_path):
    for folder_path in csv_file_path:
        for filename in os.listdir(folder_path):
            if(filename.endswith('.csv')):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                xlsx_file = file_path.replace('.csv', '.xlsx')
                df.to_excel(xlsx_file, index=False, engine='openpyxl')
    
# Funzione per calcolare la forma recente escludendo la partita in corso
import pandas as pd
def feature_engineering(folder_paths):
    global pd
# Itera su tutti i file nella cartella
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if(filename.endswith('.xlsx')):
                file_path = os.path.join(folder_path, filename)
                    # Verifica se il percorso è un file (e non una cartella)
                if os.path.isfile(file_path):
                    print(f"Processing file: {file_path}")

                data = pd.read_excel(file_path)
                # Supponiamo che il DataFrame si chiami data
                # Aggiungere le colonne per i goal cumulativi
                data['HomeGoalsCumulative'] = 0
                data['AwayGoalsCumulative'] = 0

                # Creare un dizionario per tenere traccia dei goal cumulativi di ogni squadra
                goals_cumulative = {}

                # Iterare sulle righe del DataFrame
                for index, row in data.iterrows():
                    home_team = row['HomeTeam']
                    away_team = row['AwayTeam']
                    home_goals = row['FTHG']
                    away_goals = row['FTAG']
                    
                    # Inizializzare il conteggio dei goal per le squadre se non già presente
                    if home_team not in goals_cumulative:
                        goals_cumulative[home_team] = 0
                    if away_team not in goals_cumulative:
                        goals_cumulative[away_team] = 0
                    
                    # Assegnare i goal cumulativi fino a quel momento
                    data.at[index, 'HomeGoalsCumulative'] = goals_cumulative[home_team]
                    data.at[index, 'AwayGoalsCumulative'] = goals_cumulative[away_team]
                    
                    # Aggiornare i goal cumulativi con i goal della partita attuale
                    goals_cumulative[home_team] += home_goals
                    goals_cumulative[away_team] += away_goals

                import pandas as pd

                # Supponiamo che il DataFrame si chiami data
                # Aggiungere le colonne per i punti cumulativi
                data['HomePointsCumulative'] = 0
                data['AwayPointsCumulative'] = 0

                # Creare due dizionari per tenere traccia dei punti cumulativi di ogni squadra
                points_cumulative = {}

                # Iterare sulle righe del DataFrame
                for index, row in data.iterrows():
                    home_team = row['HomeTeam']
                    away_team = row['AwayTeam']
                    result = row['FTR']
                    
                    # Inizializzare i punti per le squadre se non già presenti
                    if home_team not in points_cumulative:
                        points_cumulative[home_team] = 0
                    if away_team not in points_cumulative:
                        points_cumulative[away_team] = 0
                    
                    # Assegnare i punti cumulativi fino a quel momento
                    data.at[index, 'HomePointsCumulative'] = points_cumulative[home_team]
                    data.at[index, 'AwayPointsCumulative'] = points_cumulative[away_team]
                    
                    # Aggiornare i punti cumulativi in base al risultato della partita
                    if result == 'H':  # Vittoria della squadra di casa
                        points_cumulative[home_team] += 3
                    elif result == 'A':  # Vittoria della squadra ospite
                        points_cumulative[away_team] += 3
                    elif result == 'D':  # Pareggio
                        points_cumulative[home_team] += 1
                        points_cumulative[away_team] += 1

                # Supponiamo che il DataFrame si chiami data
                # Aggiungere le colonne per i goal subiti cumulativi
                data['HomeGoalsConcededCumulative'] = 0
                data['AwayGoalsConcededCumulative'] = 0

                # Creare un dizionario per tenere traccia dei goal subiti cumulativi di ogni squadra
                goals_conceded_cumulative = {}

                # Iterare sulle righe del DataFrame
                for index, row in data.iterrows():
                    home_team = row['HomeTeam']
                    away_team = row['AwayTeam']
                    home_goals = row['FTHG']  # Goal fatti dalla squadra di casa
                    away_goals = row['FTAG']  # Goal fatti dalla squadra ospite
                    
                    # Inizializzare i goal subiti per le squadre se non già presenti
                    if home_team not in goals_conceded_cumulative:
                        goals_conceded_cumulative[home_team] = 0
                    if away_team not in goals_conceded_cumulative:
                        goals_conceded_cumulative[away_team] = 0
                    
                    # Assegnare i goal subiti cumulativi fino a quel momento
                    data.at[index, 'HomeGoalsConcededCumulative'] = goals_conceded_cumulative[home_team]
                    data.at[index, 'AwayGoalsConcededCumulative'] = goals_conceded_cumulative[away_team]
                    
                    # Aggiornare i goal subiti cumulativi con i goal della partita attuale
                    goals_conceded_cumulative[home_team] += away_goals  # La squadra di casa subisce i goal della squadra ospite
                    goals_conceded_cumulative[away_team] += home_goals  # La squadra ospite subisce i goal della squadra di casa

                data['MatchGoal'] = data['FTHG'] + data['FTAG']

                # Calcolo della differenza di punti tra le due squadre
                data['PointsDifference'] = abs(data['HomePointsCumulative'] - data['AwayPointsCumulative'])

                # Calcolo del rapporto tra i goal fatti e subiti per la squadra di casa
                data['HomeGoalsRatio'] = data['HomeGoalsCumulative'] / data['HomeGoalsConcededCumulative']

                # Calcolo del rapporto tra i goal fatti e subiti per la squadra in trasferta
                data['AwayGoalsRatio'] = data['AwayGoalsCumulative'] / data['AwayGoalsConcededCumulative']

                # Calcolo della differenza tra i goal fatti della squadra di casa rispetto a quelli della squadra in trasferta
                # data['GoalsDifference'] = abs(data['HomeGoalsCumulative'] - data['AwayGoalsCumulative'])
                data['GoalsDifference'] = (data['HomeGoalsCumulative'] - data['AwayGoalsCumulative'])

                # Calcolo della differenza tra i goal subiti dalla squadra di casa rispetto a quelli subiti dalla squadra in trasferta
                data['ConcededGoalsDifference'] = abs(data['HomeGoalsConcededCumulative'] - data['AwayGoalsConcededCumulative'])

                data['HomePoints'] = data.apply(lambda x: 3 if x['FTHG'] > x['FTAG'] else (1 if x['FTHG'] == x['FTAG'] else 0), axis=1)
                data['AwayPoints'] = data.apply(lambda x: 3 if x['FTAG'] > x['FTHG'] else (1 if x['FTHG'] == x['FTAG'] else 0), axis=1)

                data = calculate_recent_form(data)
                data = calculate_recent_home_away_form(data)
                data = calculate_elo(data)
                data['EloRatio'] = data['elo_home']/data['elo_away']



                # Calcola i valori per ogni squadra
                for team in data['HomeTeam'].unique():
                    # Per le squadre in casa
                    home_values = calculate_last_3_matches(data, team, is_home=True)
                    data.loc[data['HomeTeam'] == team, 'HomeLast3Points'] = home_values[0]
                    data.loc[data['HomeTeam'] == team, 'HomeAvgGoalsScored'] = home_values[1]
                    data.loc[data['HomeTeam'] == team, 'HomeAvgGoalsConceded'] = home_values[2]
                    data.loc[data['HomeTeam'] == team, 'HomeEwmaPoints'] = home_values[3]
                    data.loc[data['HomeTeam'] == team, 'HomeEwmaGoalsScored'] = home_values[4]
                    data.loc[data['HomeTeam'] == team, 'HomeEwmaGoalsConceded'] = home_values[5]

                    # Per le squadre in trasferta
                    away_values = calculate_last_3_matches(data, team, is_home=False)
                    data.loc[data['AwayTeam'] == team, 'AwayLast3Points'] = away_values[0]
                    data.loc[data['AwayTeam'] == team, 'AwayAvgGoalsScored'] = away_values[1]
                    data.loc[data['AwayTeam'] == team, 'AwayAvgGoalsConceded'] = away_values[2]
                    data.loc[data['AwayTeam'] == team, 'AwayEwmaPoints'] = away_values[3]
                    data.loc[data['AwayTeam'] == team, 'AwayEwmaGoalsScored'] = away_values[4]
                    data.loc[data['AwayTeam'] == team, 'AwayEwmaGoalsConceded'] = away_values[5]

                # Calcola il numero di partite vinte, perse e pareggiate
                data['HomeWins'] = data['HomePoints'].rolling(3).apply(lambda x: (x == 3).sum()).shift(1)
                data['HomeDraws'] = data['HomePoints'].rolling(3).apply(lambda x: (x == 1).sum()).shift(1)
                data['HomeLosses'] = data['HomePoints'].rolling(3).apply(lambda x: (x == 0).sum()).shift(1)

                data['AwayWins'] = data['AwayPoints'].rolling(3).apply(lambda x: (x == 3).sum()).shift(1)
                data['AwayDraws'] = data['AwayPoints'].rolling(3).apply(lambda x: (x == 1).sum()).shift(1)
                data['AwayLosses'] = data['AwayPoints'].rolling(3).apply(lambda x: (x == 0).sum()).shift(1)

            

                # Crea la nuova colonna 'UltimoScontroDiretto'
                data['UltimoScontroDiretto'] = data.apply(lambda row: get_last_match_result(row, data), axis=1)


                data['Last3PointsDifference'] = data['HomeLast3Points'] - data['AwayLast3Points']
                data['GoalRatioDifference'] = data['HomeGoalsRatio'] - data['AwayGoalsRatio']
                data['EwmaGoalsSum'] = data['HomeEwmaGoalsScored'] + data['HomeEwmaGoalsConceded'] + data['AwayEwmaGoalsScored'] + data['AwayEwmaGoalsConceded']
                data['GoalsSum'] = data['HomeGoalsCumulative'] + data['AwayGoalsCumulative'] + data['HomeGoalsConcededCumulative'] + data['AwayGoalsConcededCumulative']
                data = add_gap_columns(data)
            
                data['GoalCumulativeSum'] = data['HomeGoalsCumulative'] + data['AwayGoalsCumulative']
                data['GoalCumulativeSumPrev'] = data['GoalCumulativeSum'].shift(1)
                data['FormRatio'] = data['HomeForm'] / data['AwayForm']
                data['FormDiff'] = abs(data['HomeForm'] - data['AwayForm'])
                data['RecentFormRatio'] = data['HomeRecentHomeForm'] / data['AwayRecentAwayForm']
                data['RecentFormDiff'] = data['HomeRecentHomeForm'] - data['AwayRecentAwayForm']
                data['UltimoScontroDiretto'] = data['UltimoScontroDiretto'].replace(
                    {'H': 1, 'A': 2, 'D': 0}).infer_objects(copy=False)
                data['EloDiff'] = abs(data['elo_home'] - data['elo_away'])
                data['isOver'] = np.where(data['MatchGoal'] > 2.5, 1, 0)

                resultFileName = 'data/engdata/'+  '/' + filename[:-5] + '-E.xlsx'
                data.to_excel(resultFileName, index=False)


def calculate_recent_form(df, window=5):
    df['HomePointsPrevious'] = df['HomePoints'].shift(1)
    df['AwayPointsPrevious'] = df['AwayPoints'].shift(1)
    
    df['HomeForm'] = df.groupby('HomeTeam')['HomePointsPrevious'].rolling(window).sum().reset_index(level=0, drop=True)
    df['AwayForm'] = df.groupby('AwayTeam')['AwayPointsPrevious'].rolling(window).sum().reset_index(level=0, drop=True)
    
    # Elimina le colonne temporanee
    df = df.drop(columns=['HomePointsPrevious', 'AwayPointsPrevious'])
    
    return df

# Funzione per calcolare la forma recente in casa e in trasferta escludendo la partita in corso
def calculate_recent_home_away_form(df, window=5):
    df['HomePointsPrevious'] = df['HomePoints'].shift(1)
    df['AwayPointsPrevious'] = df['AwayPoints'].shift(1)
    
    df['HomeRecentHomeForm'] = df.groupby('HomeTeam')['HomePointsPrevious'].rolling(window).sum().reset_index(level=0, drop=True)
    df['AwayRecentAwayForm'] = df.groupby('AwayTeam')['AwayPointsPrevious'].rolling(window).sum().reset_index(level=0, drop=True)
    
    # Elimina le colonne temporanee
    df = df.drop(columns=['HomePointsPrevious', 'AwayPointsPrevious'])
    
    return df

# Funzione per calcolare la probabilità attesa
def expected_probability(elo_a, elo_b):
    return 1 / (1 + 10**((elo_b - elo_a) / 400))

# Funzione per aggiornare il punteggio Elo
def update_elo(elo, result, expected, k=30):
    return elo + k * (result - expected)

# Funzione per calcolare il punteggio Elo per ogni partita
def calculate_elo(df, k=30, initial_elo=1500):
    # Inizializza i punteggi Elo per le squadre
    elo_dict = {}
    
    # Liste per memorizzare il punteggio Elo prima della partita
    elo_home_list = []
    elo_away_list = []
    
    # Itera attraverso ogni riga del DataFrame
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        
        # Se la squadra non ha un Elo, assegnagli l'elo iniziale
        if home_team not in elo_dict:
            elo_dict[home_team] = initial_elo
        if away_team not in elo_dict:
            elo_dict[away_team] = initial_elo
            
        # Ottieni i punteggi Elo prima della partita
        elo_home = elo_dict[home_team]
        elo_away = elo_dict[away_team]
        
        # Calcola la probabilità attesa di vittoria
        expected_home = expected_probability(elo_home, elo_away)
        expected_away = 1 - expected_home
        
        # Memorizza gli Elo attuali
        elo_home_list.append(elo_home)
        elo_away_list.append(elo_away)
        
        # Calcola il risultato della partita (1 per vittoria, 0.5 per pareggio, 0 per sconfitta)
        if row['FTHG'] > row['FTAG']:
            result_home = 1
            result_away = 0
        elif row['FTHG'] < row['FTAG']:
            result_home = 0
            result_away = 1
        else:
            result_home = 0.5
            result_away = 0.5
        
        # Aggiorna i punteggi Elo
        new_elo_home = update_elo(elo_home, result_home, expected_home, k)
        new_elo_away = update_elo(elo_away, result_away, expected_away, k)
        
        # Aggiorna il dizionario Elo
        elo_dict[home_team] = new_elo_home
        elo_dict[away_team] = new_elo_away
    
    # Aggiungi le colonne con gli Elo al DataFrame
    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    
    return df

# Funzione per calcolare i punti in base al risultato
def calculate_points(result):
    if result == 'H':
        return 3, 0
    elif result == 'A':
        return 0, 3
    else:
        return 1, 1

# Funzione per calcolare i risultati delle ultime 3 partite
def calculate_last_3_matches(df, team, is_home=True):
    points = []
    goals_scored = []
    goals_conceded = []
    
    if is_home:
        mask = (df['HomeTeam'] == team)
        points = df.loc[mask, 'HomePoints']
        goals_scored = df.loc[mask, 'FTHG']
        goals_conceded = df.loc[mask, 'FTAG']
    else:
        mask = (df['AwayTeam'] == team)
        points = df.loc[mask, 'AwayPoints']
        goals_scored = df.loc[mask, 'FTAG']
        goals_conceded = df.loc[mask, 'FTHG']
        
    # Calcolo dei punti delle ultime 3 partite
    last_3_points = points.rolling(3).sum().shift(1)
    
    # Calcolo della media dei goal fatti nelle ultime 3 partite
    avg_goals_scored = goals_scored.rolling(3).mean().shift(1)
    
    # Calcolo della media dei goal subiti nelle ultime 3 partite
    avg_goals_conceded = goals_conceded.rolling(3).mean().shift(1)
    
    # Calcolo della media esponenziale pesata dei punti nelle ultime 3 partite
    ewma_points = points.ewm(span=3).mean().shift(1)
    
    # Calcolo della media esponenziale pesata dei goal fatti nelle ultime 3 partite
    ewma_goals_scored = goals_scored.ewm(span=3).mean().shift(1)
    
    # Calcolo della media esponenziale pesata dei goal subiti nelle ultime 3 partite
    ewma_goals_conceded = goals_conceded.ewm(span=3).mean().shift(1)
    
    return last_3_points, avg_goals_scored, avg_goals_conceded, ewma_points, ewma_goals_scored, ewma_goals_conceded


# Funzione per aggiungere le colonne richieste
def add_gap_columns(df):

    # Controlla che le colonne richieste esistano
    required_columns = ['HomePointsCumulative', 'AwayPointsCumulative', 'HomeGoalsCumulative', 'AwayGoalsCumulative']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonna richiesta '{col}' non è presente nel file Excel.")
    
    # Calcola le nuove colonne
    df['HomePointGap'] = df['HomePointsCumulative'] - df['AwayPointsCumulative']
    df['AwayPointGap'] = df['AwayPointsCumulative'] - df['HomePointsCumulative']
    df['HomeGoalGap'] = df['HomeGoalsCumulative'] - df['AwayGoalsCumulative']
    df['AwayGoalGap'] = df['AwayGoalsCumulative'] - df['HomeGoalsCumulative']
    return df    

 # Funzione per ottenere l'ultimo scontro diretto
def get_last_match_result(row, data):
    # Trova tutti gli scontri diretti precedenti a quello corrente
    past_matches = data[((data['HomeTeam'] == row['HomeTeam']) & (data['AwayTeam'] == row['AwayTeam'])) |
                        ((data['HomeTeam'] == row['AwayTeam']) & (data['AwayTeam'] == row['HomeTeam']))]
    
    # Filtra solo le partite avvenute prima della partita corrente
    past_matches = past_matches[past_matches['Date'] < row['Date']]
    
    if not past_matches.empty:
        last_match = past_matches.iloc[-1]
        
        # Determina il risultato rispetto alla partita corrente
        if last_match['HomeTeam'] == row['HomeTeam']:
            return last_match['FTR']  # Il risultato dell'ultimo scontro è corretto rispetto alla partita attuale
        else:
            # Se le squadre sono invertite, inverti anche il risultato
            if last_match['FTR'] == 'H':
                return 'A'
            elif last_match['FTR'] == 'A':
                return 'H'
            else:
                return 'D'
    else:
        return None  # Se non ci sono scontri diretti precedenti, ritorna None
    

def _processData(data, filepath):
    """
    Metodo generale per elaborare i dati.
    """
    if data is None:
        data = pd.read_excel(filepath)
    data['GoalCumulativeSum'] = data['HomeGoalsCumulative'] + data['AwayGoalsCumulative']
    data['GoalCumulativeSumPrev'] = data['GoalCumulativeSum'].shift(1)
    data['FormRatio'] = data['HomeForm'] / data['AwayForm']
    data['FormDiff'] = abs(data['HomeForm'] - data['AwayForm'])
    data['RecentFormRatio'] = data['HomeRecentHomeForm'] / data['AwayRecentAwayForm']
    data['RecentFormDiff'] = data['HomeRecentHomeForm'] - data['AwayRecentAwayForm']
    data['UltimoScontroDiretto'] = data['UltimoScontroDiretto'].replace(
        {'H': 1, 'A': 2, 'D': 0}).infer_objects(copy=False)
    data['EloDiff'] = abs(data['elo_home'] - data['elo_away'])
    data['isOver'] = np.where(data['MatchGoal'] > 2.5, 1, 0)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


fullData = None


def getFullData():
    """
    Metodo specifico per elaborare i dati Full.
    """
    global fullData
    fullData = _processData(fullData, '../data/mergedFinalFull.xlsx')
    return fullData

def trainLogRegModel(x_train, y_train, class_weight=None):
    """
    Train a logistic regression model.

    Parameters
    ----------
    x_train : array-like of shape (n_samples, n_features)
        Training data.

    y_train : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    model : LogisticRegression
        Trained model.
    """
    reg = 0.01

    # train a logistic regression model on the training set
    model = LogisticRegression(C=1/reg, multi_class='ovr', class_weight=class_weight,
                               solver="liblinear", random_state=42).fit(x_train, y_train)
    return model

def class_accuracy(y_true, y_pred):
    """
    Calcola l'accuratezza per ogni classe in un modello di classificazione binaria.

    Args:
      y_true: I veri valori delle etichette.
      y_pred: I valori delle etichette predetti.

    Returns:
      Un dizionario contenente l'accuratezza per ogni classe.
    """

    cm = confusion_matrix(y_true, y_pred)

    # Calcola l'accuratezza per la classe 0
    accuracy_class_0 = cm[0, 0] / (cm[0, 0] + cm[1, 0])

    # Calcola l'accuratezza per la classe 1
    accuracy_class_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1])

    return {
        "Under": accuracy_class_0,
        "Over": accuracy_class_1
    }

def calculate_gain_ATS(row, prediction, quotaMin = 1.40):
    if (row[prediction] == 1):
        if row['QuotaATS'] > quotaMin :
            if row['FTAG'] > 0:
                return row['QuotaATS']-1
            else:
                return -1
        else:
            return 0
    else:
        return 0
    
def calculate_gain_ATW(row, prediction, quotaMin = 1.40):
    if (row[prediction] == 1):
        if row['QuotaATW'] > quotaMin :
            if row['FTR'] == 'A':
                return row['QuotaATW']-1
            else:
                return -1
        else:
            return 0
    else:
        return 0
    
def calculate_gain_ANTS(row, prediction, quotaMin = 1.40):
    if (row[prediction] == 0):
        if row['QuotaANTS'] > quotaMin :
            if row['FTAG'] == 0:
                return row['QuotaANTS']-1
            else:
                return -1
        else:
            return 0
    else:
        return 0
    
def calculate_gain_ReverseATS(row, prediction, quotaMin = 1.40):
    if (row[prediction] == 1):
        if row['QuotaANTS'] > quotaMin :
            if row['FTAG'] == 0:
                return row['QuotaANTS']-1
            else:
                return -1
        else:
            return 0
    else:
        return 0
    
def exportExcelWithTimeStamp(df, prefix, postfix):
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    df.to_excel(prefix+str(time_stamp) + postfix)