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
from collections import Iterable
import numpy as np
from operator_utils import TPOTOperatorClassFactory, Operator, ARGType
from config import classifier_config_dict_light

def flatten(tree):
    """Flatten a tree into a single, flat list."""
    for x in tree:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for y in flatten(x):
                yield y
        else:
            yield x


operators = []

classifier_config_dict_light['sklearn.preprocessing.Imputer'] = {'strategy':['median']}
classifier_config_dict_light.pop('tpot.builtins.ZeroCount', None)

for key in sorted(classifier_config_dict_light.keys()):
    op_class, arg_types = TPOTOperatorClassFactory(
        key,
        classifier_config_dict_light[key],
        BaseClass=Operator,
        ArgBaseClass=ARGType
    )
    if op_class:
        operators.append(op_class)

print(len(operators))

grammar_base = {
    'step': {
                0: { # frist step (optional, key should be a integer)
                    'input': ['input_matrix'],
                    'operator': ['Imputer']
                },
                'End': { #last step
                    'input': ['transformed_matrix', 'combine'],
                    'operator': ['roots']
                },
                'combine': { # combine step
                    'input': ['input_matrix', 'transformed_matrix', 'combine'],
                    'operator': ['CombineDFs']
                },
                'other': { # other step
                    'input': ['transformed_matrix', 'combine'],
                    'operator': ['preprocessors', 'roots']
                }
            }
    }

def generate_tree(opset, min_=1, max_=10, grammar=grammar_base, type_='main'):
    """Generate a Tree as a list of lists.

    The tree is build from the root to the leaves, and it stop growing when
    the condition is fulfilled.

    Parameters
    ----------
    opset: list
        Primitive set from which primitives are selected.
    min_: int
        Minimum height of the produced trees.
    max_: int
        Maximum Height of the produced trees.
    grammar: dictionary
        Grammar for pipeline structure.
    type_: string
        'main': main pipeline
        'branch': branch pipeline


    Returns
    -------
    individual: list
        A grown tree with leaves at possibly different depths
        dependending on the grammar
    height:
        height of tree
    """

    expr = []

    roots = [op for op in opset if op.root]
    print('Roots', len(roots))
    preprocessors = [op for op in opset if not op.root]
    print('Prepprocess', len(preprocessors))
    grammar_step = grammar['step']
    preset_step = [key for key in grammar_step.keys() if isinstance(key, int)]
    min_step = max(preset_step) + 1
    print(preset_step, min_step)

    if type_ == 'main':
        # reset min_
        min_ = max(min_, min_step)
    height = np.random.randint(min_, max_)

    if type_ == 'main'
        # last step:
        op_list = preset_step['End']['operator']
        expr.append(np.random.choice(op_list))
        remaining_height -= 1
        while remaining_height > 0:
            expr_children, branch_depth = generate_tree(opset, min_=1, max_=remaining_height, grammar=grammar_base, type_='branch')
            remaining_height -= branch_depth
            expr.append(expr_children)
    else: # 'branch'
        if height > 2:
            op_type = np.random.choice(['other', 'combine'])

    return expr, height

generate_tree(operators)
