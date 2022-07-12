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
    #     model_name="LogisticRegression", 
    #     short_window_sizes = [1,2,3,4,6],
    #     long_window_sizes = [24,42,60,180,252],
    #     periods=[1],
    #     txn_maxs=[20000,30000,100000],
    #     factors=[0.1,0.25,0.5,0.8,1,2,3]

    evaluate_model(   
        model_name=model_name, 
        short_window_sizes = [1,2,4,6],
        long_window_sizes = [24,42,60,180],
        periods=[1,2,4,6],
        txn_maxs=[20000,30000,100000],
        factors=[0.1,0.25,0.5,0.8],
        training_dataset_months=[1,3,6],
        # short_window_sizes = [2],
        # long_window_sizes = [42],
        # periods=[1],
        # txn_maxs=[100000],
        # factors=[0.5],
        save_results=True,
        save_figure=True,
        show_legend=False,
        export=False,
        save_file_prefix='',
        fee_pcts=[0.0004],
        fee_flats=[14]
    )

if __name__ == "__main__":
    fire.Fire(run)


