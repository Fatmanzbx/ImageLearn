Use Multiple models to do Image recgonition. Use Minist Dataset to test

This Package contains 3 models.
The Linear model uses soft max regression(Linear.py).
The BP Network model(CNN.py).
The Adaboost using the two above as weak classifiers.(Adaboost.py)

Firstly, the Propre.py loads data for all models with load function.
The input is to decide the location of the data and training or testing set.
Return dataset length*784 matrix for X and dataset length array for Y

Linear model: Softmax Multiple Classifier

Use Operate Linear Model.ipynb to run.

BP model: BP Network

Use BPrun.py to run, to optimize the parameter of hidden layer with Adjust function

(It runs really fast so I did not set a save function(5 iterations,1 min to reach 95%))

Adaboost model: Use BP as sub classifiers.

Use Operate Ada Boost.ipynb to run

Meanwile I wrote a CNN model, but it performs really badly and I hav not found the reason.






