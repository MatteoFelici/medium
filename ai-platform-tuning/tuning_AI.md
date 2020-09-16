# Google ML tutorials - Tuning a model with Bayesian Optimization on AI Platform
On this article of the [Google ML tutorials](https://towardsdatascience.com/tagged/google-ml-tutorials)
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