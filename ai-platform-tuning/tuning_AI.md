# Google ML tutorials - Tuning a model with Bayesian Optimization on AI Platform
On this article of the [Google ML tutorials][ML tutorials series]
series, we will talk about how to use the [AI Platform](https://cloud.google.com/ai-platform)
built-in tool to tune the hyperparameters of your Machine Learning model! We
will use a method called Bayesian Optimization to navigate the hyperparameters 
space and then find a good set.

First of all, what is Bayesian Optimization? Even if on this article we will 
focus more on the code part than on the theory behind the method, I'll try to 
give a quick overview. For a more robust and complete introduction, I suggest 
you to take a look at these articles ([1](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)
and [2](https://cloud.google.com/blog/products/gcp/hyperparameter-tuning-cloud-machine-learning-engine-using-bayesian-optimization)).

The most common ways of searching for the best hyperparameters are the **Grid Search** and the **Random Search** methods.

- In the Grid Search, the algorithm train a model for every single combination of the given hyperparameters and then returns the set with the best performance. This method is really time-consuming, especially when you want to tune more than 2-3 hyperparameters at once, because the number of models to train grows exponentially.
- In the Random Search, the algorithm instead picks at random *n* combinations of hyperparameters and train a model for each of them. Here the problem is in the *random* word: the algorithm may skip the most effective sets of hyperparameters, especially when we set a low *n*.

In a certain way, the Bayesian Optimization takes the good from both above 
methods: it does pick a subsample of all the possible combinations of 
hyperparameters, but the picking is done in a more informed way. The algorithm 
models the distribution of the objective function (say the average precision of 
our model) with a surrogate function; the domain of this function is the given 
hyperparameters space. It then explores this distribution trying different 
sets of hyperparameters. At each trial, it gains more information (in Bayes 
fashion) about the real distribution of the objective function, so it can move 
to a more "promising" subset of the domain space.

For this specific reason, keep in mind that we cannot fully parallelize the process of the Bayesian Optimization (as opposed to Grid and Random Search), since each iteration learns from the previous one.

Now let's train some models! For the tutorial, we will follow the same steps of the [training tutorial][Training article]:
- store the data on Google Storage
- write a Python application to train the model
- launch a training job on AI Platform

The big differences are on the Python application itself: we need to add a
framework to chain the model's performance results to the Bayesian Optimization. 
This framework is called [Hypertune](https://github.com/GoogleCloudPlatform/cloudml-hypertune): 
you can install it simply with `pip install cloudml-hypertune`.

## Changing the Python application
The first thing to do is to define the list of hyperparameters we want to tune. We have to train a pipeline like this

```python
pipeline = Pipeline([
    ('data_prep',
     ColumnTransformer([
        ('num_prep', StandardScaler(), num_features),
        ('cat_prep', OneHotEncoder(handle_unknown='ignore'), cat_features)
     ])),
    # ML model
    ('model',
     RandomForestClassifier(
         random_state=42,
         n_estimators=500,
         class_weight='balanced'
     ))
])
```

And maybe we want to tune some hyperparameters of the `RandomForestClassifier`:
- `max_depth`: the maximum depth of each tree of the forest
- `min_samples_split`: the minimum number (or fraction) of samples to split a node of the tree
- `max_features`: the number (or fraction) of input features to use for the training of each tree
- `max_samples`: same as `max_features`, but for the rows

To pass these hyperparameters to the application (and to the pipeline), we have to define a list of arguments with the `argparse` library, like this

```python
import argparse

...

# Instantiate an argument parser
parser = argparse.ArgumentParser()

# Define the list of input arguments
parser.add_argument('--max-depth', type=int, default=10,
                    help='Maximum depth of each tree in Random Forest model'
                         ' (integer, default 10)')
```

Then we parse the arguments and input them in the pipeline

```python
# Parse arguments
args = parser.parse_args()

...

pipeline = Pipeline([
    ('data_prep',
     ColumnTransformer([
        ('num_prep', StandardScaler(), num_features),
        ('cat_prep', OneHotEncoder(handle_unknown='ignore'), cat_features)
     ])),
    # ML model
    ('model',
     RandomForestClassifier(
         random_state=42,
         n_jobs=args.n_jobs,
         n_estimators=args.n_estimators,
         max_depth=args.max_depth,
         max_features=args.max_features,
         min_samples_split=args.min_samples_split,
         class_weight='balanced',
         max_samples=args.max_samples
     ))
])
```

After that, we need a strategy to assess the performance for each set of given hyperparameters. 
We use the **cross-validation** methodology:
1. you divide your data into *n* splits
2. choose one split as *validation*
3. concatenate the remaining *n-1* splits and train the model on this new dataset
4. calculate the performance on the hold-out split
5. repeat 2-4 on each split

This method is suitable if you want to robustly assess one single model, because you train and validate it on *n* potentially different scenarios.

We can use the pre-built `cross_validate` [function](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)
 from scikit-learn.

```python
from sklearn.model_selection import cross_validate

...

scores = cross_validate(pipeline, train, y_train,
                        scoring=['accuracy', 'precision', 'recall', 'f1'],
                        cv=5)
```

We provide:
- a valid classifier (we can use a model like `RandomForestClassifier`, in our case the above-defined pipeline)
- input data and target
- one or more metrics to calculate (like accuracy and precision) - [here](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
 the full list of available metrics
- a value for *n* (the `cv` parameter)

The `scores` result is a dictionary with an entry for each given metric. For example, 
`scores['test_accuracy']` will be a vector with the 5 calculated accuracies on the 5 iterations.

Finally, we have to use the `hyperopt` framework. Since the whole optimization is 
based on a single value, we have choose one particular metric (*F1-score*) and 
compute the average value.

```python
# Instantiate a hypertune object
hpt = hypertune.HyperTune()

# Compute the average metric
avg_f1 = scores['test_f1'].mean()

# Pass the value to hyperopt
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='F1',
    metric_value=avg_f1,
    global_step=1
)
``` 

And that's it for the Python application! 

But hey! We have defined the hyperparameters to tune, but not which values (or 
range of values) the application has to try with the Bayesian Optimization. How 
can we do this?


## Specify hyperparameters - The *config* file 


[ML tutorials series]: https://towardsdatascience.com/tagged/google-ml-tutorials
[Training article]: https://towardsdatascience.com/training-a-model-on-google-ai-platform-84ceff87b5f3