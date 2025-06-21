import os
import sys
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import evaluate_models, save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier,
)




class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise CustomException(e,sys)
        



         
        
    def train_model(self,X_train,y_train,x_test,y_test):

            models = {
                "LogisticRegression": LogisticRegression(verbose=1, random_state=42),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
                "RandomForestClassifier": RandomForestClassifier(verbose=1, random_state=42),
                "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1, random_state=42),
            }

            params = {
                "LogisticRegression": {
                    'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['saga', 'liblinear'],      # saga supports all penalties, liblinear supports l1/l2
                    'max_iter': [100, 200, 500],
                    'class_weight': [None, 'balanced']
                },
                "DecisionTreeClassifier": {
                    'criterion': ['gini','entropy','log_loss'],
                    'splitter': ['best','random'],
                    'max_features': ['sqrt','log2', None],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced']
                },
                "RandomForestClassifier": {
                    'n_estimators': [100, 200, 500],
                    'criterion': ['gini','entropy','log_loss'],
                    'max_features': ['sqrt','log2', None],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced', 'balanced_subsample']
                },
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1, 10],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
                "GradientBoostingClassifier": {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'subsample': [0.5, 0.75, 1.0],
                    'max_depth': [3, 5, 7],
                    'max_features': ['sqrt', 'log2', None],
                    'loss': ['log_loss', 'deviance']
                }
            }
            # Train the models
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train
                                              ,X_test=x_test,y_test=y_test
                                              ,models=models,param=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            y_train_pred=best_model.predict(X_train)
            classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

          

            




            y_test_pred=best_model.predict(x_test)
            classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
            
           
            
            preprocessor=load_object(file_path=self.data_transformation_artifact.tranfsormed_object_file_path)
            model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            Network_model=NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
            
            save_object("final_model/model.pkl",best_model)


            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                 train_metric_artifact=classification_train_metric,
                                 test_metric_artifact=classification_test_metric)
            logging.info(f"Model Trainer artifact:{model_trainer_artifact}")
            return  model_trainer_artifact


    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            model=self.train_model(x_train,y_train,x_test,y_test)
            return model
        except Exception as e:
            raise CustomException(e,sys)
