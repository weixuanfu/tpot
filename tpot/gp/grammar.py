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
import re
import importlib
from copy import deepcopy

import numpy as np
from sklearn.pipeline import make_pipeline

from .util import flatten
from .individual import Individual
from ..builtins import StackingEstimator


def is_estimator(model):
    """Determine if a class is a machine-learning estimator."""
    return 'predict' in dir(model)


def import_module(module_path):
    """Return an imported class from a path."""
    *path, class_name = module_path.split('.')
    path = '.'.join(path)
    model_class = getattr(importlib.import_module(path), class_name)

    return model_class


class Grammar(object):
    """A machine-learning pipeline grammar."""

    _variable_re = re.compile(r'^\$(\S+)$')
    _starting_rule = 'make_pipeline'
    _ctx_base = {
        'make_pipeline': make_pipeline,
        'StackingEstimator': StackingEstimator
    }
    _grammar_base = {
        'make_pipeline': {('make_pipeline', '(', '$pipeline', ')')},
        'pipeline': {('$ops', '$est')},
        'ops': {('$prep', '$ops')} | {('$est_transform', '$ops')} | {()},
        'combine': {('make_union', '(', '$make_pipeline', ', ', '$make_pipeline', ')')},
        'est': set(),
        'prep': set(),
        'est_transform': {('StackingEstimator', '(', 'estimator=', '$est', '), ')},
    }

    def __init__(self, rules, ctx):
        """Instantiate a Grammar.

        Parameters
        ----------
        rules : dict
            The grammar rules.
        ctx : dict
            The evaluation context for individuals in the grammar.
        """
        self._rules = rules
        self.ctx = ctx

    @classmethod
    def from_config(cls, config):
        """Instantiate a Grammar from a TPOT config dictionary.

        Parameters
        ----------
        config : dict
            A TPOT operator configuration dictionary.
        """
        grammar = deepcopy(cls._grammar_base)
        ctx = deepcopy(cls._ctx_base)

        for module_path, parameters in config.items():
            Grammar._add_model(module_path, parameters, grammar, ctx)

        return Grammar(grammar, ctx)

    @staticmethod
    def _add_model(module_path, parameters, grammar, ctx):
        """Add a model and its parameters to a grammar.

        Parameters
        ----------
        module_path : str
            The path to a model class.
        parameters : dict
            A dictionary relating parameter names (str) to values (Iterable)
        grammar : dict
            Grammar rules to add to.
        ctx : dict
            Evaluation context for the grammar.
        """
        try:
            model_class = import_module(module_path)
        except Exception as e:
            # Could not import model
            return

        model_name = model_class.__name__
        # Add the model to our evaluation context
        ctx[model_name] = model_class

        param_steps = [Grammar._add_parameter(name, values, model_name, grammar) for name, values in parameters.items()]
        model_entry = {(model_name, '(', *list(flatten(param_steps)), '), ')}
        model_type = 'est' if is_estimator(model_class) else 'prep'

        # Add model to the set of estimators or preprocessors
        grammar[model_type] |= model_entry

    @staticmethod
    def _add_parameter(param_name, param_values, model_name, grammar):
        """Add a model's parameters to a grammar.

        If param_values contains no items, None will be added to the set of
        parameter values.

        Parameters
        ----------
        param_name : str
            Name of a parameter for a model.
        param_values : Iterable
            All possible values for the parameter.
        model_name : str
            The name of the model the parameter belongs to.
        grammar : dict
            Grammar rules to add the parameter to.

        Returns
        -------
        grammar steps that connect the model to the parameter, e.g:

        Existing grammar:
        {
            'DecisionTreeClassifier': {('DecisionTreeClassifier', '(', ..., ')')}
        }

        Return value:
        ['max_depth', '=', '$DecisionTreeClassifier__max_depth', ', ']
        """
        grammar_param_name = '{}__{}'.format(model_name, param_name)
        grammar[grammar_param_name] = {(val.__repr__(), ) for val in param_values}

        return [param_name, '=', '${}'.format(grammar_param_name), ', ']

    def reverse(self, start=None):
        """Reverse a CFG to produce a random word in its language.

        Parameters
        ----------
        start : str
            The starting rule in the grammar.

        Returns
        -------
        An Individual from a word in the CFG.
        """
        if start is None:
            start = self._starting_rule

        tree = []
        rule = self._rules[start]

        # Pick a random path to branch into, then add the selected branch onto
        # the tree, recursing as needed.
        for atom in np.random.choice(tuple(rule)):
            groups = re.findall(self._variable_re, str(atom))

            if len(groups) == 0:
                tree.append(str(atom))
            else:
                nested_tree = self.reverse(groups[0])
                tree.append(nested_tree)

        return Individual(tree, self)

    def DFS(self, max_depth=6, start=None):
        """Yield nodes in the grammar through depth-first search.

        Parameters
        ----------
        max_depth : int
            The maximum depth to perform in the search.
        start : str
            The starting rule in the grammar.

        Returns
        -------
        A generator yielding nodes in the grammar.
        """
        if max_depth < 1:
            raise RecursionError

        if start is None:
            start = self._starting_rule

        rule = self._rules[start]

        for branch in rule:
            for atom in branch:
                groups = re.findall(self._variable_re, str(atom))

                if len(groups) == 0:
                    yield atom
                else:
                    yield self.BFS(max_depth - 1, groups[0])
