To run:

1. Make sure you have numpy and skikit-learn
2. run python proj.py

Details:
-round1() is the hyperparameter step. It will output the parameters, mean, and standard deviation for each parameter combination in order of score rank based on the cv_results_ for each
-round2() handles the performance comparision for each. It will output the scores as well as the mean and standard deviation for each. The parameters are preset for each and will need to be manually changed if different parameters are to be used.
-load_data() will handle loading the data from the file and is pulled from past homeworks.
-output_results(cv_results) just cleans up the default output from GridSearchCV

Note for reproducing:
I used np.random.seed(1) before going along with each step of the experiment which show hopefully make it reproducable.

All data can be found in the data folder

round1() and round2() operate independently from one another so feel free to comment them out in the main function
