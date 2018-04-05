#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 2018

@author: grixor
"""
import numpy as np
import pandas as pd
import os, os.path
from sklearn.base import BaseEstimator, TransformerMixin

from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split


class TPOT_Data_Selector(BaseEstimator):
    # def __init__(self, subset_dir=None, expert_source=None, ekf_index=None, k_best=None, contain_header=True):

    def __init__(self, subset_dir=None, generations=100, population_size=100,
                 offspring_size=None, mutation_rate=0.9, crossover_rate=0.1,
                 scoring=None, cv=5, subsample=1.0, n_jobs=1,
                 max_time_mins=None, max_eval_time_mins=5,
                 random_state=None, config_dict=None,
                 warm_start=False, memory=None,
                 periodic_checkpoint_folder=None, early_stop=None,
                 verbosity=0, disable_update_check=False):
        """Set up the subset selector for pipeline optimization.

        Parameters
        ----------
        subset_dir: directory, required
            Path to folder that stores the feature list files. Currently,
            each file needs to be a .csv with one header row. The feature
            names in these files must match those in the (training and
            testing) dataset.
        generations: int, optional (default: 100)
            Number of iterations to the run pipeline optimization process.
            Generally, TPOT will work better when you give it more generations (and
            therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        population_size: int, optional (default: 100)
            Number of individuals to retain in the GP population every generation.
            Generally, TPOT will work better when you give it more individuals
            (and therefore time) to optimize the pipeline. TPOT will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        offspring_size: int, optional (default: None)
            Number of offspring to produce in each GP generation.
            By default, offspring_size = population_size.
        mutation_rate: float, optional (default: 0.9)
            Mutation rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the GP algorithm how many pipelines to apply random
            changes to every generation. We recommend using the default parameter unless
            you understand how the mutation rate affects GP algorithms.
        crossover_rate: float, optional (default: 0.1)
            Crossover rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the genetic programming algorithm how many pipelines to
            "breed" every generation. We recommend using the default parameter unless you
            understand how the mutation rate affects GP algorithms.
        scoring: string or callable, optional
            Function used to evaluate the quality of a given pipeline for the
            problem. By default, accuracy is used for classification problems and
            mean squared error (MSE) for regression problems.

            Offers the same options as sklearn.model_selection.cross_val_score as well as
            a built-in score 'balanced_accuracy'. Classification metrics:

            ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc']

            Regression metrics:

            ['neg_median_absolute_error', 'neg_mean_absolute_error',
            'neg_mean_squared_error', 'r2']

            If you would like to use a custom scoring function, you can pass a callable
            function to this parameter with the signature scorer(y_true, y_pred).
            See the section on scoring functions in the documentation for more details.

            TPOT assumes that any custom scoring function with "error" or "loss" in the
            name is meant to be minimized, whereas any other functions will be maximized.
        cv: int or cross-validation generator, optional (default: 5)
            If CV is a number, then it is the number of folds to evaluate each
            pipeline over in k-fold cross-validation during the TPOT optimization
             process. If it is an object then it is an object to be used as a
             cross-validation generator.
        subsample: float, optional (default: 1.0)
            Subsample ratio of the training instance. Setting it to 0.5 means that TPOT
            randomly collects half of training samples for pipeline optimization process.
        n_jobs: int, optional (default: 1)
            Number of CPUs for evaluating pipelines in parallel during the TPOT
            optimization process. Assigning this to -1 will use as many cores as available
            on the computer.
        max_time_mins: int, optional (default: None)
            How many minutes TPOT has to optimize the pipeline.
            If provided, this setting will override the "generations" parameter and allow
            TPOT to run until it runs out of time.
        max_eval_time_mins: int, optional (default: 5)
            How many minutes TPOT has to optimize a single pipeline.
            Setting this parameter to higher values will allow TPOT to explore more
            complex pipelines, but will also allow TPOT to run longer.
        random_state: int, optional (default: None)
            Random number generator seed for TPOT. Use this parameter to make sure
            that TPOT will give you the same results each time you run it against the
            same data set with that seed.
        config_dict: a Python dictionary or string, optional (default: None)
            Python dictionary:
                A dictionary customizing the operators and parameters that
                TPOT uses in the optimization process.
                For examples, see config_regressor.py and config_classifier.py
            Path for configuration file:
                A path to a configuration file for customizing the operators and parameters that
                TPOT uses in the optimization process.
                For examples, see config_regressor.py and config_classifier.py
            String 'TPOT light':
                TPOT uses a light version of operator configuration dictionary instead of
                the default one.
            String 'TPOT MDR':
                TPOT uses a list of TPOT-MDR operator configuration dictionary instead of
                the default one.
            String 'TPOT sparse':
                TPOT uses a configuration dictionary with a one-hot-encoder and the
                operators normally included in TPOT that also support sparse matrices.
        warm_start: bool, optional (default: False)
            Flag indicating whether the TPOT instance will reuse the population from
            previous calls to fit().
        memory: a Memory object or string, optional (default: None)
            If supplied, pipeline will cache each transformer after calling fit. This feature
            is used to avoid computing the fit transformers within a pipeline if the parameters
            and input data are identical with another fitted pipeline during optimization process.
            String 'auto':
                TPOT uses memory caching with a temporary directory and cleans it up upon shutdown.
            String path of a caching directory
                TPOT uses memory caching with the provided directory and TPOT does NOT clean
                the caching directory up upon shutdown.
            Memory object:
                TPOT uses the instance of sklearn.external.joblib.Memory for memory caching,
                and TPOT does NOT clean the caching directory up upon shutdown.
            None:
                TPOT does not use memory caching.
        periodic_checkpoint_folder: path string, optional (default: None)
            If supplied, a folder in which tpot will periodically save the best pipeline so far while optimizing.
            Currently once per generation but not more often than once per 30 seconds.
            Useful in multiple cases:
                Sudden death before tpot could save optimized pipeline
                Track its progress
                Grab pipelines while it's still optimizing
        early_stop: int or None (default: None)
            How many generations TPOT checks whether there is no improvement in optimization process.
            End optimization process if there is no improvement in the set number of generations.
        verbosity: int, optional (default: 0)
            How much information TPOT communicates while it's running.
            0 = none, 1 = minimal, 2 = high, 3 = all.
            A setting of 2 or higher will add a progress bar during the optimization procedure.
        disable_update_check: bool, optional (default: False)
            Flag indicating whether the TPOT version checker should be disabled.

        Returns
        -------
        None

        """
        self.subset_dir = subset_dir

        # ---------------------------------------------------------------------#
        # The following code is copied from Class TPOTBase of base.py:
        # ---------------------------------------------------------------------#
        if self.__class__.__name__ == 'TPOTBase':
            raise RuntimeError('Do not instantiate the TPOTBase class directly; use TPOTRegressor or TPOTClassifier instead.')

        # Prompt the user if their version is out of date
        self.disable_update_check = disable_update_check
        if not self.disable_update_check:
            update_check('tpot', __version__)

        self._pareto_front = None
        self._optimized_pipeline = None
        self._optimized_pipeline_score = None
        self._exported_pipeline_text = ""
        self.fitted_pipeline_ = None
        self._fitted_imputer = None
        self._imputed = False
        self._pop = []
        self.warm_start = warm_start
        self.population_size = population_size
        self.generations = generations
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.max_eval_time_seconds = max(int(self.max_eval_time_mins * 60), 1)
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.early_stop = early_stop
        self._last_optimized_pareto_front = None
        self._last_optimized_pareto_front_n_gens = 0
        self.memory = memory
        self._memory = None # initial Memory setting for sklearn pipeline

        # dont save periodic pipelines more often than this
        self._output_best_pipeline_period_seconds = 30

        # Try crossover and mutation at most this many times for
        # any one given individual (or pair of individuals)
        self._max_mut_loops = 50

        # Set offspring_size equal to population_size by default
        if offspring_size:
            self.offspring_size = offspring_size
        else:
            self.offspring_size = population_size

        self.config_dict_params = config_dict
        self._setup_config(self.config_dict_params)

        self.operators = []
        self.arguments = []
        for key in sorted(self.config_dict.keys()):
            op_class, arg_types = TPOTOperatorClassFactory(
                key,
                self.config_dict[key],
                BaseClass=Operator,
                ArgBaseClass=ARGType
            )
            if op_class:
                self.operators.append(op_class)
                self.arguments += arg_types

        # Schedule TPOT to run for many generations if the user specifies a
        # run-time limit TPOT will automatically interrupt itself when the timer
        # runs out
        if max_time_mins is not None:
            self.generations = 1000000

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        if self.mutation_rate + self.crossover_rate > 1:
            raise ValueError(
                'The sum of the crossover and mutation probabilities must be <= 1.0.'
            )

        self.verbosity = verbosity
        self.operators_context = {
            'make_pipeline': make_pipeline,
            'make_union': make_union,
            'StackingEstimator': StackingEstimator,
            'FunctionTransformer': FunctionTransformer,
            'copy': copy
        }
        self._pbar = None
        # Specifies where to output the progress messages (default: sys.stdout).
        # Maybe open this API in future version of TPOT.(io.TextIOWrapper or io.StringIO)
        self._file = sys.stdout

        # Dictionary of individuals that have already been evaluated in previous
        # generations
        self.evaluated_individuals_ = {}
        self.random_state = random_state

        self._setup_scoring_function(scoring)

        self.cv = cv
        self.subsample = subsample
        if self.subsample <= 0.0 or self.subsample > 1.0:
            raise ValueError(
                'The subsample ratio of the training instance must be in the range (0.0, 1.0].'
            )
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self._setup_pset()
        self._setup_toolbox()
        # ---------------------------------------------------------------------#
        # end copy
        # ---------------------------------------------------------------------#


    def get_subset(self, input_data, input_target):
        self.input_data = input_data
        self.input_target = input_target
        self.feature_names = list(self.input_data.columns.values)
        
        self.subset_files = os.listdir(self.subset_dir)
        self.num_subset = len(self.subset_files)
        self.feature_set = {}
        self.data_subset = {}
        self.population_size = population_size

        for i in range(self.num_subset):
            self.subset_i = self.subset_dir + "/" + self.subset_files[i]
            self.features_i_df = pd.read_csv(self.subset_i, sep='\t', header=0)
            # what if not csv? what other file types we should support?
            self.feature_i = set(features_i_df.values.flatten())
            self.feature_set[i] = list(feature_i.intersection(set(self.feature_names)))
            self.data_subset[i] = self.input_data[self.feature_set[i]]

        return self


    def fit(self, input_data, input_target, train_size=0.75, test_size=0.25):
    """Fit an optimized machine learning pipeline using TPOT.

    Uses genetic programming to optimize a machine learning pipeline that
    maximizes score on the provided features and target. Performs internal
    k-fold cross-validaton to avoid overfitting on the provided data. The
    best pipeline is then trained on the entire set of provided samples.

    Parameters
    ----------
    features: array-like {n_samples, n_features}
        Feature matrix

        TPOT and all scikit-learn algorithms assume that the features will be numerical
        and there will be no missing values. As such, when a feature matrix is provided
        to TPOT, all missing values will automatically be replaced (i.e., imputed) using
        median value imputation.

        If you wish to use a different imputation strategy than median imputation, please
        make sure to apply imputation to your feature set prior to passing it to TPOT.
    target: array-like {n_samples}
        List of class labels for prediction
    sample_weight: array-like {n_samples}, optional
        Per-sample weights. Higher weights force TPOT to put more emphasis on those points
    groups: array-like, with shape {n_samples, }, optional
        Group labels for the samples used when performing cross-validation.
        This parameter should only be used in conjunction with sklearn's Group cross-validation
        functions, such as sklearn.model_selection.GroupKFold

    Returns
    -------
    self: object
        Returns a copy of the fitted TPOT object

    """
        # how do I pass tpot parameters here
        # self.fit(input_data, input_target)
        self.input_target = input_target

        # if input_target is discrete  ----- pseudocode
        pipeline_dict = {}
        for i in range(self.num_subset):
            #X_train, X_test, y_train, y_test = train_test_split(self.data_subset[i], input_target,
                                                                #train_size, test_size)
            subX suby!
            pipeline_optimizer = TPOTClassifier(generations=5, population_size=self.population_size,
                                                cv=5, verbosity=2)
            pipeline_optimizer.fit(X_train, y_train)
            pipeline_dict[subset_idx] = pipeline_optimizer.fitted_pipeline_
        return pipeline_dict
            #pipeline_optimizer.export('tpot_exported_pipeline_' + i + '.py')

        # else:
        # for i in range(self.num_subset):
        #     X_train, X_test, y_train, y_test = train_test_split(self.data_subset[i], input_target,
        #                                                         train_size, test_size)
        #     pipeline_optimizer = TPOTRegressor(generations=5, population_size=20,
        #                                         cv=5, verbosity=2)
        #     pipeline_optimizer.fit(X_train, y_train)
        #     pipeline_optimizer.export('tpot_exported_pipeline_' + i + '.py')
