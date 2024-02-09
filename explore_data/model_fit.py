#model_fit.py
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import (roc_curve, auc, classification_report, confusion_matrix, roc_auc_score)

PATH_TO_MODELS = '../../artifacts/artifacts_models/'

class ModelFit:
    '''
    Fit, cross-validate, predict, and evaluate the performance of a scikit-learn predictor.

    Attributes:
        path_to_models (str): The path to the directory where model artifacts are stored.

    Methods:
        - fit_model(classifier, X_train, y_train, seed=42): Fit a scikit-learn predictor.
        - cross_validate_model(X_train, y_train, seed=42): Cross-validate the fitted predictor.
        - predict_and_evaluate(X_val, y_val, threshold, target_names): Predict probabilities and evaluate predictions.
        - manual_experiment_tracking(classifiers, threshold, X_train, y_train, X_val, y_val, experiment_descriptor):
          Return performance of selected models and save results to a CSV file.
        - train_n_plot(estimator, X_input, Y_input, feature_groups, group_names=[], include_intercept_coef_feature_importance=True):
          Train and plot ROC curves and display relevant model parameters (intercept, coefficients, or feature importances).
        - format_non_scientific(df): Format numeric columns in a DataFrame to non-scientific notation with two decimal places.

    Args:
        path_to_models (str, optional): The path to the directory where model artifacts are stored (default: '../../artifacts/artifacts_models/').

    Methods:
        - fit_model(classifier: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> None:
          Fit a scikit-learn predictor.

        - cross_validate_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> None:
          Cross-validate the fitted scikit-learn predictor.

        - predict_and_evaluate(X_val: pd.DataFrame, y_val: pd.Series, threshold: float, target_names: list) -> None:
          Predict probabilities of observations belonging to the positive class and evaluate predictions.

        - manual_experiment_tracking(classifiers: dict, threshold: float, X_train: pd.DataFrame, y_train: pd.Series,
          X_val: pd.DataFrame, y_val: pd.Series, experiment_descriptor: str) -> None:
          Return performance of selected models and save results to a CSV file.

        - train_n_plot(estimator: BaseEstimator, X_input: pd.DataFrame, Y_input: pd.Series, feature_groups: List[List[str]],
          group_names: List[str] = [], include_intercept_coef_feature_importance: bool = True) -> Union[Dict[int, Tuple[Dict[str, Union[np.ndarray, float]], Union[np.ndarray, None]]]:
          Train and plot ROC curves for feature groups and display relevant model parameters (intercept, coefficients, or feature importances).

        - format_non_scientific(df: pd.DataFrame) -> pd.DataFrame:
          Format numeric columns in a DataFrame to non-scientific notation with two decimal places.
    '''
    def __init__(self, path_to_models: str = PATH_TO_MODELS):
        self.fitted_model = None
        self.path_to_models = path_to_models

    def fit_model(self, classifier: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> None:
        '''
        Fit a `scikit-learn` predictor.

        Args:
            classifier (BaseEstimator): `scikit-learn` predictor to fit.
            X_train (pd.DataFrame): Training set.
            y_train (pd.Series): Labels of the training set.
            seed (int): Random seed for reproducibility (default: 42).
        '''
        self.fitted_model = classifier.fit(X_train, y_train)

    def cross_validate_model(self, X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> None:
        '''
        Cross-validate the fitted `scikit-learn` predictor.

        Args:
            X_train (pd.DataFrame): Training set.
            y_train (pd.Series): Labels of the training set.
            seed (int): Random seed for reproducibility (default: 42).
        '''
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted. Call 'fit_model' first.")

        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        scores = cross_val_score(
            self.fitted_model,
            X_train,
            y_train,
            cv=stratified_kfold,
            scoring='roc_auc',
        )
        print(f"Validation score: \n{np.round(scores, 5)}")
        print(f"\nAverage score: {round(scores.mean(), 2)}")
        print(f"\nStandard deviation: {round(scores.std(), 2)}")

    def predict_and_evaluate(self, X_val: pd.DataFrame, y_val: pd.Series, threshold: float, target_names: list) -> None:
        '''
        Predict probabilities of observations belonging to the positive class and evaluate predictions.

        Args:
            X_val (pd.DataFrame): Validation set.
            y_val (pd.Series): Labels of the validation set.
            threshold (float): Threshold for classification.
            target_names (list): List of target class names.

        Raises:
            ValueError: If the model has not been fitted. Call 'fit_model' first.
        '''
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted. Call 'fit_model' first.")

        soft_preds = self.fitted_model.predict_proba(X_val)[:, 1]
        thresholded_preds = (soft_preds > threshold).astype(int)
        class_report = classification_report(y_val, thresholded_preds, target_names=target_names)
        print("Classification Report - validation set:\n", class_report)
        print("\n")
        cf_matrix = confusion_matrix(y_val, thresholded_preds)
        print("Confusion matrix:\n", cf_matrix)
        print("\n")
        roc_auc = roc_auc_score(y_val, thresholded_preds)
        print("roc_auc_score:\n", round(roc_auc, 3))
        print("\n")

    def manual_experiment_tracking(
        self,
        classifiers: dict,
        threshold: float,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        experiment_descriptor: str
    ) -> None:
        '''
        Return performance of selected models and save results to a CSV file.

        Args:
            classifiers (dict): Dictionary of classifier names and instances.
            threshold (float): Threshold for classification.
            X_train (pd.DataFrame): Training set.
            y_train (pd.Series): Labels of the training set.
            X_val (pd.DataFrame): Validation set.
            y_val (pd.Series): Labels of the validation set.
            experiment_descriptor (str): Descriptor for the experiment results file.
        '''
        results_df = pd.DataFrame(columns=['Classifier', 'ROC_AUC'])

        for name, clf in classifiers.items():
            fitted_model = clf.fit(X_train, y_train)
            soft_preds = fitted_model.predict_proba(X_val)[:, 1]
            thresholded_preds = (soft_preds > threshold).astype(int)

            class_report = classification_report(y_val, thresholded_preds, target_names=['0', '1'])
            print(f'Classification Report, {name} - validation set:\n', class_report)
            print('\n')

            roc_score_result = roc_auc_score(y_val, thresholded_preds)
            results_df = results_df.append({'Classifier': name, 'ROC_AUC': roc_score_result}, ignore_index=True)

        results_df = results_df.sort_values(by='ROC_AUC', ascending=False)
        results_df.to_csv(f'{self.path_to_models}/{experiment_descriptor}.csv', index=False)
        print(results_df)

    def _train_group(self, estimator: BaseEstimator, X: pd.DataFrame, Y: pd.Series):
        '''
        Train a group of features with a given estimator and return ROC metrics.

        Args:
            estimator (BaseEstimator): The estimator to train.
            X (pd.DataFrame): Input features for the group.
            Y (pd.Series): Target labels.

        Returns:
            tprs (list): True positive rates for each fold.
            aucs (list): AUC scores for each fold.
            mean_fpr (np.ndarray): Mean false positive rate.
        '''
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        cv = StratifiedKFold(n_splits=5)
        for train, test in cv.split(X, Y):
            probas_ = estimator.fit(X[train], Y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        return tprs, aucs, mean_fpr

    def _plot_group(self, group_name: str, tprs: list, aucs: list, mean_fpr: np.ndarray):
        '''
        Plot ROC curves and AUC values for a group.

        Args:
            group_name (str): Name of the feature group.
            tprs (list): True positive rates for each fold.
            aucs (list): AUC scores for each fold.
            mean_fpr (np.ndarray): Mean false positive rate.
        '''
        fig = plt.figure(1, figsize=(16, 7))
        fig.set_dpi(100)
        fig.suptitle(group_name)

        ax0 = plt.axes([0.05, 0.05, 0.15, 0.9])

        i = 0
        for tpr, auc in zip(tprs, aucs):
            ax0.plot(mean_fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, auc))
            i += 1

        ax0.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax0.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=0.7)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax0.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r'$\pm$ 1 std. dev.')

        ax0.set_title("ROC Curve")
        ax0.legend(loc="lower right")

    def train_n_plot(self, estimator: BaseEstimator, X_input: pd.DataFrame, Y_input: pd.Series, feature_groups: List[List[str]], group_names: List[str] = [],
                 include_intercept_coef_feature_importance: bool = True) -> Tuple[Dict[int, Dict[str, Union[np.ndarray, float]]], Union[np.ndarray, None]]:
        '''
        Train the estimator on feature groups, plot ROC curves, and display relevant model parameters (intercept, coefficients, or feature importances).
        Args:
        estimator (BaseEstimator): A scikit-learn estimator for classification.
        X_input (pd.DataFrame): Input data as a DataFrame.
        Y_input (pd.Series): Target data as a Series.
        feature_groups (List[List[str]]): List of lists where each sublist contains feature names to be used in a group.
        group_names (List[str], optional): List of group names (default: []).
        include_intercept_coef_feature_importance (bool, optional): Whether to include model parameters (intercept, coefficients, or feature importances) (default: True).

        Returns:
        Tuple[Dict[int, Dict[str, Union[np.ndarray, float]]], Union[np.ndarray, None]]: A dictionary containing results for each group
        and optionally the model parameters (intercept and coefficients or feature importances).
        '''
        group_count = len(feature_groups)
        results = dict()
        model_params = None

        for counter in range(group_count):
            features = feature_groups[counter]
            group_name = group_names[counter] if counter < len(group_names) else "(" + ", ".join(features) + ")"

            # Ensure X_selected and Y_selected are DataFrames
            X_selected = X_input.loc[X_input.index.isin(Y_input.index), features]
            Y_selected = Y_input[X_input.index.isin(Y_input.index)]

            X_updated = X_selected.values
            Y_updated = Y_selected.values

            tprs, aucs, mean_fpr = self._train_group(estimator, X_updated, Y_updated)

            self._plot_group(group_name, tprs, aucs, mean_fpr)

            if include_intercept_coef_feature_importance:
                if hasattr(estimator, 'intercept_') and hasattr(estimator, 'coef_'):
                    intercept = estimator.intercept_
                    coef = estimator.coef_
                    model_params = {'intercept': intercept, 'coefficients': coef}
                elif hasattr(estimator, 'feature_importances_'):
                    feature_importance = estimator.feature_importances_
                    model_params = {'feature_importance': feature_importance}

            results[counter] = model_params

        return results, model_params