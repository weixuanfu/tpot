```python
from tpot import TPOTRegressor
from sklearn.datasets import load_iris # a classification benchmark with 0,1,2 outcome
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
# exclude 2 outcome to make binary classification problem
binary_index = np.where(iris.target != 2)

X = iris.data[binary_index[0],:]
y = iris.target[binary_index]
print("Shape of X",X.shape)

print("Shape of y", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)


print(X_train[:5,])

```

    Shape of X (100, 4)
    Shape of y (100,)
    [[5.7 4.4 1.5 0.4]
     [5.  3.5 1.3 0.3]
     [5.7 2.9 4.2 1.3]
     [4.9 3.1 1.5 0.1]
     [6.3 2.5 4.9 1.5]]
    


```python
print(y_train[:5,])
```

    [0 0 1 0 1]
    


```python
print(X_train.shape)
```

    (75, 4)
    


```python
y_adj_train = y_train[:5]-X_train[:5, -1]
# assume the last column of X is pi
print("Max and Min value of y_adj", max(y_adj_train),min(y_adj_train))
print(np.mean(y_train[:5]))
```

    Max and Min value of y_adj -0.1 -0.5
    0.4
    


```python
# I assume the last column of X_train or X_test is pi, below is the scoring function for our design. 
```


```python
def custom_reg_scoring_function(estimator, X, y):
    from tpot.metrics import balanced_accuracy
    pi = X[:,-1] # pi
    X_no_pi = X[:,:-1] # X without pi
    y_adj = y - pi # y-pi
    estimator.fit(X_no_pi, y_adj) # fit X_no_pi and y_adj to regression pipeline
    y_adj_pred = estimator.predict(X_no_pi) # get prediction of y_adj
    y_adj_pred_pi = y_adj_pred + X[:, -1] # add pi back to prediction of y_adj
    # make a array of 0 for redefined prediction of y
    pred_y = np.zeros(y_adj_pred_pi.shape, dtype=int)
    # assume that y_adj_pred_pi > 0.5 then pred_y is 1 unless it is 0
    pred_y[np.where(y_adj_pred_pi > 0.5)] = 1
    return balanced_accuracy(y, pred_y)
    
```


```python
tpot = TPOTRegressor(generations=5, population_size=50, scoring=custom_reg_scoring_function, verbosity=3, random_state=42)
tpot.fit(X_train, y_train)

```

    29 operators have been imported by TPOT.
    


    A Jupyter Widget


    _pre_test decorator: _random_mutation_operator: num_test=0 Found array with 0 feature(s) (shape=(50, 0)) while a minimum of 1 is required..
    _pre_test decorator: _random_mutation_operator: num_test=0 Found array with 0 feature(s) (shape=(50, 0)) while a minimum of 1 is required..
    _pre_test decorator: _random_mutation_operator: num_test=0 Unsupported set of arguments: The combination of penalty='l2' and loss='epsilon_insensitive' are not supported when dual=False, Parameters: penalty='l2', loss='epsilon_insensitive', dual=False.
    _pre_test decorator: _random_mutation_operator: num_test=1 Expected n_neighbors <= n_samples,  but n_samples = 50, n_neighbors = 65.
    Pipeline encountered that has previously been evaluated during the optimization process. Using the score from the previous evaluation.
    Pipeline encountered that has previously been evaluated during the optimization process. Using the score from the previous evaluation.
    Generation 1 - Current Pareto front scores:
    -1	1.0	RandomForestRegressor(input_matrix, RandomForestRegressor__bootstrap=True, RandomForestRegressor__max_features=0.7500000000000001, RandomForestRegressor__min_samples_leaf=11, RandomForestRegressor__min_samples_split=9, RandomForestRegressor__n_estimators=100)
    
    _pre_test decorator: _random_mutation_operator: num_test=0 Found array with 0 feature(s) (shape=(50, 0)) while a minimum of 1 is required..
    _pre_test decorator: _random_mutation_operator: num_test=0 Unsupported set of arguments: The combination of penalty='l2' and loss='epsilon_insensitive' are not supported when dual=False, Parameters: penalty='l2', loss='epsilon_insensitive', dual=False.
    Generation 2 - Current Pareto front scores:
    -1	1.0	RandomForestRegressor(input_matrix, RandomForestRegressor__bootstrap=True, RandomForestRegressor__max_features=0.7500000000000001, RandomForestRegressor__min_samples_leaf=11, RandomForestRegressor__min_samples_split=9, RandomForestRegressor__n_estimators=100)
    
    Generation 3 - Current Pareto front scores:
    -1	1.0	RandomForestRegressor(input_matrix, RandomForestRegressor__bootstrap=True, RandomForestRegressor__max_features=0.7500000000000001, RandomForestRegressor__min_samples_leaf=11, RandomForestRegressor__min_samples_split=9, RandomForestRegressor__n_estimators=100)
    
    _pre_test decorator: _random_mutation_operator: num_test=0 Unsupported set of arguments: The combination of penalty='l2' and loss='epsilon_insensitive' are not supported when dual=False, Parameters: penalty='l2', loss='epsilon_insensitive', dual=False.
    _pre_test decorator: _random_mutation_operator: num_test=0 Found array with 0 feature(s) (shape=(50, 0)) while a minimum of 1 is required..
    Generation 4 - Current Pareto front scores:
    -1	1.0	RandomForestRegressor(input_matrix, RandomForestRegressor__bootstrap=True, RandomForestRegressor__max_features=0.7500000000000001, RandomForestRegressor__min_samples_leaf=11, RandomForestRegressor__min_samples_split=9, RandomForestRegressor__n_estimators=100)
    
    Pipeline encountered that has previously been evaluated during the optimization process. Using the score from the previous evaluation.
    Pipeline encountered that has previously been evaluated during the optimization process. Using the score from the previous evaluation.
    Pipeline encountered that has previously been evaluated during the optimization process. Using the score from the previous evaluation.
    Generation 5 - Current Pareto front scores:
    -1	1.0	RandomForestRegressor(input_matrix, RandomForestRegressor__bootstrap=True, RandomForestRegressor__max_features=0.7500000000000001, RandomForestRegressor__min_samples_leaf=11, RandomForestRegressor__min_samples_split=9, RandomForestRegressor__n_estimators=100)
    
    




    TPOTRegressor(config_dict=None, crossover_rate=0.1, cv=5,
                  disable_update_check=False, early_stop=None, generations=5,
                  max_eval_time_mins=5, max_time_mins=None, memory=None,
                  mutation_rate=0.9, n_jobs=1, offspring_size=None,
                  periodic_checkpoint_folder=None, population_size=50,
                  random_state=42,
                  scoring=<function custom_reg_scoring_function at 0x00000277945746A8>,
                  subsample=1.0, template=None, use_dask=False, verbosity=3,
                  warm_start=False)




```python
tpot.score(X_test, y_test)
```




    1.0




```python
tpot.fitted_pipeline_
```




    Pipeline(memory=None,
             steps=[('randomforestregressor',
                     RandomForestRegressor(bootstrap=True, criterion='mse',
                                           max_depth=None,
                                           max_features=0.7500000000000001,
                                           max_leaf_nodes=None,
                                           min_impurity_decrease=0.0,
                                           min_impurity_split=None,
                                           min_samples_leaf=11, min_samples_split=9,
                                           min_weight_fraction_leaf=0.0,
                                           n_estimators=100, n_jobs=None,
                                           oob_score=False, random_state=None,
                                           verbose=0, warm_start=False))],
             verbose=False)




```python

```
