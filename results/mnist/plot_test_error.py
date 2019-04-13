import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sys

sns.set()

dropout_path = "Dropout_baseline/"
sgd_path = "SGD_baseline/"
bayesian_path = "bayes1/"

dropout_error = [round(100-accu, 2) for accu in pickle.load(open(dropout_path+"test_accu_lst.pkl", "rb"))]
sgd_error = [round(100-accu, 2) for accu in pickle.load(open(sgd_path+"test_accu_lst.pkl", "rb"))]
bayesian_error = [round(100-accu, 2) for accu in pickle.load(open(bayesian_path+"test_accu_lst.pkl", "rb"))]

plt.plot(dropout_error, label="Dropout")
plt.plot(sgd_error, label="Vanilla SGD")
plt.plot(bayesian_error, label="Bayes by Backprop")
plt.ylim(0.8, 2.3)
plt.xlabel("Epochs")
plt.ylabel("Test error (%)")

plt.legend()
plt.show()
