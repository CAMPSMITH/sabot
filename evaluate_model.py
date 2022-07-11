"""evaluate_model
A command line tool to evaluate a model
Example:
    $ python evaluate_model.py SVC
"""
import sys
import fire
from utils.eval import evaluate_model

suported_models = ['SVC',
                        'LogisticRegression',
                        'DecisionTreeClassifier',
                        'GradientBoostingClassifier',
                        'AdaBoostClassifier',
                        'GaussianNB']
                        
def run(model_name):
    assert model_name in suported_models, f"{model_name} is not valid"
    evaluate_model(   
    model_name="LogisticRegression", 
    short_window_sizes = [2],
    long_window_sizes = [42],
    periods=[1],
    txn_maxs=[100000],
    factors=[0.5],
    save_results=True,
    save_figure=True,
    save_file_prefix='fee_',
    fee_pcts=[0, 0.000125, 0.001, 0.00125, 0.0015, 0.00175],
    fee_flats=[0, 13, 80, 100, 125, 130, 135, 140, 145]
)

if __name__ == "__main__":
    fire.Fire(run)


