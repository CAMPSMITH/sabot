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
    evaluate_model(model_name)

if __name__ == "__main__":
    fire.Fire(run)


