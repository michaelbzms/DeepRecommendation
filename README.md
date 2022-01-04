# DeepRecommendation

In this project I explore the usage of Deep Learning methods for recommendation, mainly Neural Collaborative Filtering with modifications from other methods.

[//]: # (### Recommendation and Neural Collaborative Filtering)

Suppose we have access to **users**, **items** and **recorded interactions** between them. 

Given a user **u** and an item **i**, the goal of a recommender system is to predict a preference score **f**(**u**, **i**) between user **u** and item **i**.

The goal in Neural Collaborative Filtering is to learn that function **f** from recorded user-item interactions using a **neural network**. 
The simplest architecture that I found to achieve that is shown below and is an extension of **matrix factorization** methods.

![BasicNeuralCollaborativeFiltering](neural_collaborative_filtering.PNG)

Given appropriate user and item vectors representing **u** and **i** (e.g. one-hot vectors, feature vectors, etc), 
we embed them into their own embedding space using a separate projection layer for each (these are learnt during training), 
then we concatenate the two resulting item and user embeddings and pass them into a multi-layer perceptron (aka Neural CF Layers)
until we get the predicted preference score for the pair. The loss could be MSE if we are doing e.g. rating regression or cross entropy if e.g. we have binary interactions.

Beyond this architecture I experiment with the following:
- Using **content-based methods** for creating item profiles from relevant item features and user profiles from aggregated item profiles. This makes the learned function **f** applicable to new items and new users but requires relevant item features that are problem-specific.
- Using an **item-item attention mechanism** to differently weight items associated with a user when creating a user profile from item profiles.
- Representing user-item interactions in a bipartite graph and using **Graph Neural Networks** to learn user and item embeddings directly from the graph instead of implicitly from one user-item pair at a time.


[//]: # (### Utility Matrix)

[//]: # ()
[//]: # (User-item interactions &#40;e.g. views, likes, ratings&#41; are typically stored in a matrix called **utility matrix**, like the one showed below. )

[//]: # ()
[//]: # (![UtilityMatrixExample]&#40;utility_matrix_example.PNG&#41;)

[//]: # ()
[//]: # (On one dimension there are items, on the other there are the users. The cell values can be anything denoting *preference*, *dispreference* or *lack there of both*.)

[//]: # ()
[//]: # (In this formulation, the goal of a recommender system is to predict the blanks i.e. the unknown interactions. )

[//]: # (Note that predicting a cell of the utility matrix in this setting is equivalent to calling **f**&#40;u, i&#41; for the appropriate user-item pair &#40;u, i&#41;.)

## Project Organization

I split the code in two main parts:
1. `neural_collaborative_filtering/`
2. the rest

Everything in the `neural_collaborative_filtering/` folder is meant to be very generic and flexible 
so that it can be easily applied to any similarly formulated problem simply by extending some classes. 
Basically, you need to extend and implement an appropriate dataset class in `datasets` 
and pick an appropriate model for it or implement your own after extending one in `models/base.py`.
Training and evaluating logic is implemented once for all different models in `train.py` and `eval.py` by 
"pushing" differences in calling different `forward()`s in the dataset class.


Everything else is problem-specific to my task of movie recommendations and can serve as a usage example.
