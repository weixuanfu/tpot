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
import numpy as np
from sklearn.externals.joblib import Parallel, delayed


def chunks(iterable, chunk_size):
    """Yield chunks of iterable of size n."""
    for x in range(0, len(iterable), chunk_size):
        yield iterable[x:x + chunk_size]


class Population(object):
    """A population of sklearn pipelines."""

    def __init__(self, grammar, logger):
        """Instantiate a Population.

        Parameters
        ----------
        grammar : Grammar
            A grammar from which Individuals will be generated.
        """
        self._grammar = grammar
        self._logger = logger
        self._pop = []
        self._n_gen = 0  # Generation number
        # All generated Individual objects, keyed by their pipeline string
        self._individual_cache = {}

    def starting_generation(self, population_size):
        """Generate a new population.

        Parameters
        ----------
        population_size : int
            The population size.
        """
        self._pop = [self.generate_individual() for _ in range(population_size)]

    def generate_individual(self):
        """Return a random Individual.

        If multiple identical individuals are produced through this method, the
        subsequent individuals will be pulled from a cache. This prevents
        identical individuals from being evaluated twice.
        """
        individual = self._grammar.reverse()
        pipeline_str = str(individual)

        try:
            # If the instance already exists in the cache, return that
            return self._individual_cache[pipeline_str]
        except KeyError:
            # Add this individual to the dictionary and return it
            self._individual_cache[pipeline_str] = individual
            return individual

    def evaluate_population(self, features, target, cv, scoring_function, pbar, n_jobs, sample_weight=None, groups=None):
        """Determine the fit of the provided individuals.

        Parameters
        ----------
        features: numpy.ndarray {n_samples, n_features}
            A numpy matrix containing the training and testing features for
            the individual's evaluation.
        target: numpy.ndarray {n_samples}
            A numpy matrix containing the training and testing target for the
            individual's evaluation.
        cv:
            Cross-validation generator.
        scoring_function: callable
            The scoring function.
        pbar: tqdm progress bar
            The progress bar object.
        n_jobs: int
            Number of threads to run.
        sample_weight: array-like {n_samples}, optional
            List of sample weights to balance (or un-balanace) the dataset
            target as needed.
        groups: array-like {n_samples, }, optional
            Group labels for the samples used while splitting the dataset
            into train/test set.

        Returns
        -------
        scores: list of floats
            Returns a list of tuple value indicating the individual's fitness
            according to its performance on the provided data
        """
        eval_args = (features, target, cv, scoring_function, sample_weight, groups)

        # Don't use parallelization if n_jobs == 1
        if n_jobs == 1:
            scores = self._linear_eval(*eval_args)
        else:
            scores = self._parallel_eval(n_jobs, *eval_args)

        for index, score in enumerate(scores):
            self._assign_score(index, score)

        return scores

    def _assign_score(self, index, score):
        """Assign a score to an individual in the population.

        Parameters
        ----------
        index : int
            Index of the individual in self._pop.
        score : any
            Score returned by the scoring function.
        """
        individual = self._pop[index]

        if score == "Timeout":
            individual.score = -float('inf')
        elif not isinstance(individual.score, (float, np.float64, np.float32)):
            raise ValueError('Scoring function did not return a float.')
        else:
            individual.score = score

    def _parallel_eval(self, n_jobs, *eval_args):
        """Evaluate the population with parallelization.

        Parameters
        ----------
        n_jobs : int
            Number of threads to run.
        eval_args : vararg
            Arguments to apply to the evaluation of each individual.

        Returns
        -------
        Generator yielding each score.
        """
        # Run a series of batches, with 4 pipelines per thread
        for chunk in chunks(self._pop, chunk_size=n_jobs * 4):
            parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')
            chunk_scores = parallel(delayed(ind.cv_evalulate)(*eval_args) for ind in chunk)

            for score in chunk_scores:
                self._logger.update_pbar(1)
                yield score

    def _linear_eval(self, *eval_args):
        """Evaluate the population in a single thread..

        Parameters
        ----------
        eval_args : vararg
            Arguments to apply to the evaluation of each individual.

        Returns
        -------
        Generator yielding each score.
        """
        for ind in self._pop:
            self._logger.update_pbar(1)
            yield ind.cv_evalulate(*eval_args)

    def apply_mutations(self, lambda_, cx_prob, mut_prob):
        """Apply mutations to the population."""
        pass

    def ea_mu_plus_lambda(self, mu, lambda_, cx_prob, mut_prob):
        """Perform a generation of growth on the population using eaMuPlusLambda."""
        pass
