# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from finta import TA
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Import a new classifiers from SKLearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import classification_report_imbalanced

from utils.utils import epoch_to_datetime
from utils.trade import add_class_labels, backtest_model
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
    
def get_cryptocompare_data(file_path):
    """ Prep the data by creating a simple ohlcv df, inddex Time(UTC)
    """
    df = pd.read_csv(
        file_path,
        infer_datetime_format=True,
        parse_dates=True,
        index_col='Time (UTC)'
    )
    df.drop(columns=['volume_from'], inplace = True)
    df.rename(columns = {'volume_to':'volume'}, inplace = True)
    return df

def get_kucoin_data(file_path):
    """ Prep the data by creating a simple ohlcv df, inddex Time(UTC)
    """    
    df = pd.read_csv(
        file_path,
        infer_datetime_format=True,
        parse_dates=True,
    )
    df['epoch'] = df['epoch']/1000  # from epoch in ms to epoch in seconds
    df['Time (UTC)'] = df['epoch'].apply(epoch_to_datetime) 
    df.drop(columns=['epoch'], inplace = True)
    df.set_index('Time (UTC)', inplace = True)
    return df

def add_engineered_features(df, fast_sma_window, slow_sma_window):
    """ assumes that the df is a standard simlpe ohlcv df
    
        Engineered features include:
        * fast SMA for close
        * slow SMA for close
        * fast SMA for volume
        * slow SMA for volume
            
    """
    df['SMA_fast_close'] = df['close'].rolling(window=fast_sma_window).mean()
    df['SMA_slow_close'] = df['close'].rolling(window=slow_sma_window).mean()
    df['SMA_fast_volume'] = df['volume'].rolling(window=fast_sma_window).mean()
    df['SMA_slow_volume'] = df['volume'].rolling(window=slow_sma_window).mean()
    ohlc = df.drop(columns=['volume'])
    # df['Bollinger Bands'] = TA.BBANDS(df.drop(columns=['volume'],MA=TA.KAMA(100)))
    bol_df = TA.BBANDS(ohlc)
    df = pd.concat([df,bol_df],axis=1)
    # df['Donchian Channel'] = TA.DO(
    #     ohlc = ohlc,
    #     upper_period = slow_sma_window,
    #     lower_period = fast_sma_window    
    # )  # list index out of range
    # df['Directional Movement Indicator'] = TA.DMI(df)
    df['Mass Index'] = TA.MI(df)
    apz_df = TA.APZ(df)
    apz_df.columns=['APZ_UPPER','APZ_LOWER']
    df = pd.concat([df,apz_df],axis=1)
    return df


def create_train_test_datasets(months, df):
    """ calculate the trainind start and end based on the given df.  df is assumed to have a column y that denotes the class labels
    """
    
    # separate df into X and y
    df = df.dropna()
    y = df['y']
    X = df.drop(columns=['y'])
    
    # separate dataset into training and testing dataset based on months
    training_begin = X.index.min()
    training_end = X.index.min() + DateOffset(months=months)

    # create the training features dataset X_train and training classigication labels y_train for the training timeframe
    X_train = X.loc[training_begin:training_end]
    y_train = y.loc[training_begin:training_end]

    # create the testing features dataset X_test and testing classigication labels y_test following the training timeframe
    X_test = X.loc[training_end+DateOffset(hours=1):]
    y_test = y.loc[training_end+DateOffset(hours=1):]
    
    # Use StandardScaler to scale the data.
    # Scale the features DataFrames

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Apply the scaler model to fit the X-train data
    X_scaler = scaler.fit(X_train)

    # Transform the X_train and X_test DataFrames using the X_scaler
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)  
    return X_train_scaled, y_train, X_test_scaled, y_test, X_scaler


def evaluate_model(model_name,training_dataset_months=[8],periods=[1]):
    print(f"start:  {datetime.now()}")    

    sources = [
        {"provider":"cryptocompare","label":"BTC","path":Path('data/BTC_historical_price.csv')},
        # {"provider":"cryptocompare","label":"DAI","path":Path('data/DAI_historical_price.csv')},
        {"provider":"cryptocompare","label":"ETH","path":Path('data/ETH_historical_price.csv')},
        # {"provider":"cryptocompare","label":"USDC","path":Path('data/USDC_historical_price.csv')},
        {"provider":"cryptocompare","label":"USDT","path":Path('data/USDT_historical_price.csv')},
        {"provider":"kucoin","label":"sUSD/USDT","path":Path('data/sUSD_USDC_ku_historical_price.csv')}
    ]

    # factors = [0.1, 0.25, 0.5, 0.8, 1, 2, 3]
    factors = [0.8,1]
    # short_window_sizes = [1,2,3,4,6]
    short_window_sizes = [1]
    # long_window_sizes = [24,42,60,180,252]
    long_window_sizes = [42]
    # txn_maxs = [20000,30000,100000]
    txn_maxs = [100000]

    # define a dictionary of models to train
    alternate_models = {}

    # Initiate the model instances

    alternate_models['SVC'] = {
            "model": svm.SVC()
        }
    alternate_models['LogisticRegression'] = {
            "model": LogisticRegression()
        }
    alternate_models['DecisionTreeClassifier'] = {
            "model": DecisionTreeClassifier()
        }
    alternate_models['GradientBoostingClassifier'] = {
            "model": GradientBoostingClassifier()
        }
    alternate_models['AdaBoostClassifier'] = {
            "model": AdaBoostClassifier()
        }
    alternate_models['GaussianNB'] = {
            "model": GaussianNB()
        }

    models = {}
    max_return = None
    selected_model = None

    permutation_count = 0

    # construct the model permutations and evaluate the model permutations
    results_df = pd.DataFrame()
    returns = None
    max_return = 0
    max_f1 = 0
    selected_return_model = None
    selected_f1_model = None
    for training_months in training_dataset_months:
        for fast_sma_window in short_window_sizes:
            for slow_sma_window in long_window_sizes:
                for period in periods:
                    for factor in factors:
                        for txn_max in txn_maxs:
                            # create a key for this model permutation
                            model_key = f"{model_name}-p{period}-tr({training_months})-sw({fast_sma_window})-lw({slow_sma_window})-fa({factor})-max({txn_max})"
                            permutation_count += 1
                            # configure model permutation
                            models[model_key] = {
                                "model_name":model_name,
                                "training_months":training_months,
                                "fast_sma_window":fast_sma_window,
                                "slow_sma_window":slow_sma_window,
                                "model":alternate_models[model_name]["model"],
                            }

                            # get source data, for each source data, get engineered features
                            datasets = []
                            for source in sources:
                                df = None
                                if source['provider']=='cryptocompare':
                                    df = get_cryptocompare_data(source['path'])
                                elif source['provider']=='kucoin':
                                    df = get_kucoin_data(source['path'])

                                # add engineereg features derived from ohlcv
                                df = add_engineered_features(df,fast_sma_window,slow_sma_window)

                                # prefix column names so they can be differentiated
                                cols = df.columns
                                new_cols = []
                                for col in cols:
                                    new_cols.append(f"{source['label']}_{col}")                                        
                                df.columns=new_cols

                                datasets.append(df)
                                # df = df.add_class_labels(df,slow_sma_window,factor,period)

                            # concatenate all of the coin and associated
                            df = pd.concat(datasets,axis=1)

                            # add class labels for training
                            df = add_class_labels(df,slow_sma_window,factor,period)

                            # add actual returns
                            df['returns'] = df['sUSD/USDT_close'].pct_change()

                            # create training and testing datasets
                            models[model_key]['X_train_scaled'], models[model_key]['y_train'], models[model_key]['X_test_scaled'], models[model_key]['y_test'], models[model_key]['x_scaler'] = create_train_test_datasets(training_months,df)

                            # use random oversampling to address class imbalance
                            random_oversampler = RandomOverSampler()
                            models[model_key]['X_train_scaled'], models[model_key]['y_train'] = random_oversampler.fit_resample(models[model_key]['X_train_scaled'], models[model_key]['y_train'])            

                            # train model
                            models[model_key]['trained_model'] =  models[model_key]['model'].fit(models[model_key]['X_train_scaled'], models[model_key]['y_train'])

                            # get predictions
                            y_predictions = models[model_key]['trained_model'].predict( models[model_key]['X_test_scaled'])
                            # print(f"# of y_test: {len(models[model_key]['y_test'])}    # of predictions: {len(y_predictions)}")
                            models[model_key]['y_predictions'] = pd.DataFrame(
                                {
                                    "y_test":models[model_key]['y_test'],
                                    "y_prediction":y_predictions,
                                    "Actual Returns":df.loc[models[model_key]['y_test'].index.min():models[model_key]['y_test'].index.max():,'returns'],
                                    "close":df.loc[models[model_key]['y_test'].index.min():models[model_key]['y_test'].index.max():,'sUSD/USDT_close'],
                                    "future close":df.loc[models[model_key]['y_test'].index.min():models[model_key]['y_test'].index.max():,'sUSD/USDT_future']
                                })

                            models[model_key]['y_predictions'].dropna()

                            # Classification reports
                            models[model_key]['classification_report'] = classification_report_imbalanced(
                                models[model_key]['y_predictions']['y_test'], 
                                models[model_key]['y_predictions']['y_prediction'],
                                output_dict=True
                            )

                            # Print the classification report
                            # print(f"{model_key}:\n{models[model_key]['classification_report']}") 
                            # {-1: {
                            #     'pre': 0.6790799561883899, 
                            #     'rec': 0.2806699864191942, 
                            #     'spe': 0.9420719652036378, 
                            #     'f1': 0.39718129404228053, 
                            #     'geo': 0.5142094181164019, 
                            #     'iba': 0.24692310827785746, 
                            #     'sup': 2209}, 
                            #  0: {
                            #      'pre': 0.4516821760916249, 
                            #      'rec': 0.884992987377279, 
                            #      'spe': 0.3060022650056625, 
                            #      'f1': 0.5981042654028436, 
                            #      'geo': 0.5203939456330897, 
                            #      'iba': 0.2864894982201783, 
                            #      'sup': 2852}, 
                            #  1: {
                            #      'pre': 0.779373368146214, 
                            #      'rec': 0.27062556663644605, 
                            #      'spe': 0.9666073898439044, 
                            #      'f1': 0.4017496635262449, 
                            #      'geo': 0.5114574005638033, 
                            #      'iba': 0.2433825764634188, 
                            #      'sup': 2206}, 
                            #  'avg_pre': 0.620281111815607, 
                            #  'avg_rec': 0.514792899408284, 
                            #  'avg_spe': 0.6998887206449207, 
                            #  'avg_f1': 0.47742212759146885, 
                            #  'avg_geo': 0.515801178369128, 
                            #  'avg_iba': 0.2613765183415491, 
                            #  'total_support': 7267}      

                            f1_score = 1-( \
                                (1 - models[model_key]['classification_report'][-1]['f1'])**2 + \
                                (1 - models[model_key]['classification_report'][0]['f1'])**2 + \
                                (1 - models[model_key]['classification_report'][1]['f1'])**2 )
                                
                            if f1_score > max_f1:
                                max_f1 = f1_score
                                selected_f1_model = model_key
            
                            # backtest model
                            models[model_key]['backtest'] = backtest_model(
                                model_key,
                                models[model_key]['y_predictions'],
                                txn_max = txn_max
                            )

                            # add the cumulative return to the list of returns for plotting
                            # add the actual and signal returns if the 
                            if returns is None:
                                # This is the permutation for the model
                                # create the returns dataframe and add the actual returns and the signal returns
                                returns = {
                                    "actual": models[model_key]['backtest']['actual_cum_return']
                                    # "strategy": (1 +models[model_key]['backtest']['Strategy Returns']).cumprod()
                                }
                            returns[model_key] = models[model_key]['backtest']['strategy_cum_return']
                            cum_return = returns[model_key].iloc[-1]
                            print(f"{model_key} f1 score: {f1_score} cumulative return: {cum_return}")

                            if cum_return > max_return:
                                max_return = cum_return
                                selected_return_model = model_key

                            df_dictionary = pd.DataFrame([{
                                "permutation": model_key,
                                "model": model_name,
                                "f1 score": f1_score,
                                "strategy return": cum_return,
                                "period": period,
                                "training months": training_months,
                                "fast window": fast_sma_window,
                                "slow window": slow_sma_window,
                                "factor": factor,
                                "max transaction": txn_max 
                            }])
                            results_df = pd.concat([results_df, df_dictionary], ignore_index=True)
                    
    # create a plot for the family of returns for the range of training monts, and SMA window sizes
    returns_df = pd.DataFrame(returns)
    model_family_plot = returns_df.plot(
        figsize=(10,7),
        title=f'{model_name} Cumulative Returns for various training and SMA window sizes'
    )
    
    # save plot
    model_family_plot.figure.savefig(f'images/{model_name}_returns.png', bbox_inches='tight')

    # save final results for the family
    results_df.to_csv(Path(f"results/{model_name}.csv"), index=False)

    # show the max return achieved with the model
    print(f"maximum cumulative return for {model_name} models was {max_return} from model permutation {selected_return_model}")
    print(f"maximum f1 for {model_name} models was {max_f1} from model permutation {selected_f1_model}")

    # print number of permutations tested
    print(f"{permutation_count} permutations tested ")

    print(f"end:  {datetime.now()}")