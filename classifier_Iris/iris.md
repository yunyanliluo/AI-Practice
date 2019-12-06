### *IRIS Classifier*

#### Three IRIS classifiers without optimizers

- classifier1_hw.py

  My homework code. It uses a two-layer NN for training.

  Its procedure is described below:

  1. Load data from file system and pre-process data
  2. Shuffle data and generate dataset
  3. Generate trainable variables, train 200 epochs and test in every epoch
  4. Draw loss and accuracy curves


- classifier2_standard_reference.py

  A standard reference. 

  Comparing to classifier1_hw.py, it adds data processing functions including  normalize(), norm_nonlinear() and standardize(). Besides, it uses a one-layer dense NN for training. It is appropriate for the IRIS dataset to use a small dense NN. One layer is enough. 

- classifier3_load_online.py

  A simplier version provided by my teacher. It loads IRIS data from online sources.

However, classifier1_hw.py needs to improve. The dense layer needs an extra activation function, while the output layer only needs linear processing.  

#### Other classifiers with optimizers

These are classifiers with optimizers including SGD, SGD(momentum), adam, adagrad, adadelta. 