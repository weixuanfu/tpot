# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import inspect
import warnings

import numpy as np
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._split import check_cv
from sklearn.utils import indexable
from sklearn.metrics.scorer import check_scoring
from sklearn.base import clone, is_classifier
from stopit import threading_timeoutable, TimeoutException

from .grammar import flatten


def pipeline_models(pipeline):
    """Yield all model classes in a pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A sklearn Pipeline.

    Returns
    -------
    Generator yielding all models in a pipeline.
    """
    def pipeline_iter(pipeline_steps):
        """Yield each model in the pipeline."""
        recursive_attrs = ['steps', 'transformer_list', 'estimators']

        for (_, step) in pipeline_steps:
            for attr in recursive_attrs:
                if hasattr(step, attr):
                    for x in pipeline_iter(getattr(step, attr)):
                        yield x
                    break
            else:
                yield step

    return pipeline_iter(pipeline.steps)


def set_attr_recursive(pipeline, attribute, value):
    """Recurse through all objects in a pipeline and set an attribute to a value.

    Parameters
    ----------
    pipeline: Pipeline
        A sklearn pipeline.
    attribute: str
        The parameter to assign a value for in each pipeline object
    value: any
        The value to assign the parameter to in each pipeline object
    """
    for model in pipeline_models(pipeline):
        if hasattr(model, attribute):
            setattr(model, attribute, value)


class Individual(object):
    """A machine learning pipeline."""

    def __init__(self, parse_tree, rules_used, grammar, random_state=None):
        """Instantiate a new Individual from a parse tree.

        Parameters
        ----------
        parse_tree : list
            A parse-tree generated from a pipeline grammar.
        grammar : Grammar
            The grammar from which the Individual was generated.
        random_state : tuple
            The starting PRNG state the Individual was generated from.
        """
        self._parse_tree = parse_tree
        self._rules_used = rules_used
        self._grammar = grammar
        self._starting_prng_state = random_state
        self._sklearn = None
        self.score = None
        self.compexity = self.operator_count

    def __str__(self):
        flattened = list(flatten(self._parse_tree))
        return ''.join(flattened)

    def to_sklearn(self, seed=42):
        """Convert the individual to a sklearn pipeline.

        Parameters
        ----------
        seed : int (optional)
            Seed for models in the pipeline.

        Returns
        -------
        sklearn Pipeline
        """
        if self._sklearn is None:
            pipeline_string = str(self)
            self._sklearn = eval(pipeline_string, self._grammar.ctx)

            if seed is not None:
                # Fix random state when the operator allows
                set_attr_recursive(self._sklearn.steps, 'random_state', seed)

                if 'XGBClassifier' in self.model_names or 'XGBRegressor' in self.model_names:
                    # Setting the seed is needed for XGBoost support because XGBoost
                    # stores both a seed and random_state, and they're not synced
                    # correctly. XGBoost will raise an exception if
                    # random_state != seed.
                    set_attr_recursive(self._sklearn.steps, 'seed', seed)

        return self._sklearn

    @property
    def operator_count(self):
        """Count the number of steps in the sklearn pipeline."""
        return len(list(pipeline_models(self.to_sklearn())))

    @property
    def model_names(self):
        """Return a list of the names of all models used."""
        models = list(pipeline_models(self.to_sklearn()))
        model_names = [model.__name__ for model in models]

        return model_names

    def set_sample_weight(self, sample_weight=None):
        """Recursively iterates through all objects in the pipeline and sets sample weight.

        Parameters
        ----------
        sample_weight: array-like
            List of sample weight

        Returns
        -------
        A dictionary of sample_weight
        """
        sample_weight_dict = {}

        if sample_weight is not None:
            for model in pipeline_models(self.to_sklearn()):
                if inspect.getargspec(model.fit).args.count('sample_weight'):
                    model_sw = model.__name__ + '__sample_weight'
                    sample_weight_dict[model_sw] = sample_weight

        return sample_weight_dict

    @threading_timeoutable(default="Timeout")
    def cv_evalulate(self, features, target, cv, scoring_function, sample_weight=None, groups=None):
        """Fit estimator and compute scores for a given dataset split.

        Parameters
        ----------
        features : array-like of shape at least 2D
            The data to fit.
        target : array-like, optional, default: None
            The target variable to try to predict in the case of
            supervised learning.
        cv: int or cross-validation generator
            If CV is a number, then it is the number of folds to evaluate each
            pipeline over in k-fold cross-validation during the TPOT optimization
            process. If it is an object then it is an object to be used as a
            cross-validation generator.
        scoring_function : callable
            A scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
        sample_weight : array-like, optional
            List of sample weights to balance (or un-balanace) the dataset
            target as needed.
        groups: array-like {n_samples, }, optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        Cross-validation score.
        """
        # Don't evaluate the pipeline if it's already been scored
        if self.score is not None:
            return self.score

        if self.model_names.count('PolynomialFeatures') > 1:
            return -float('inf')

        cv = check_cv(cv, target, classifier=is_classifier(self.to_sklearn()))
        cv_iter = list(cv.split(features, target, groups))

        def score(train, test):
            """Score the individual on a training/testing data pair.

            Parameters
            ----------
            train : array-like
                Training features and target.
            test : array-like
                Testing features and target.
            """
            indx_features, indx_target = indexable(features, target)
            scorer = check_scoring(self.to_sklearn(), scoring=scoring_function)

            return _fit_and_score(
                estimator=clone(self.to_sklearn()),
                X=indx_features,
                y=indx_target,
                scorer=scorer,
                train=train,
                test=test,
                verbose=0,
                parameters=None,
                fit_params=self.set_sample_weight(sample_weight)
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                scores = np.array([score(train, test) for train, test in cv_iter])[:, 0]
                return np.mean(scores)
        except TimeoutException:
            return "Timeout"
        except Exception as e:
            return -float('inf')

    def random_mutation(self):
        """Perform a random mutation from possible mutation operators."""
        return np.random.choice([
            self.point_mutation,
            self.insert_mutation,
            self.shrink_mutation
        ])()

    def point_mutation(self):
        """Perform a point mutation.

        Returns
        -------
        A new Individual.
        """
        pass

    def insert_mutation(self):
        """Perform an insert mutation.

        Returns
        -------
        A new Individual.
        """
        pass

    def shrink_mutation(self):
        """Perform a shrink mutation.

        Returns
        -------
        A new Individual.
        """
        # Don't perform a shrink if it would create an empty pipeline
        if len(self.model_names) > 1:
            pass

    def crossover_with(self, other):
        """Perform crossover with another individual.

        Parameters
        ----------
        other : Individual
            A different individual to perform the crossover with.

        Returns
        -------
        A new Individual.
        """
        pass
