import numpy as np

def multiclass_svm(scores, y_index):
    scores = scores - scores[y_index] + 1
    return np.sum(scores, axis=1)

def multiclass_softmax(scores, y_index):
    scores -= np.max(scores, axis=1)
    scores = np.exp(scores)
    normal = np.sum(scores, axis=1)
    return -scores[:,y_index] / normal


a = np.array([[10, -2 ,3]])
print multiclass_svm(a, 0)
a = np.array([[10, -2 ,3]])
print multiclass_softmax(a, 0)

a = np.array([[10, 9, 9]])
print multiclass_svm(a, 0)
a = np.array([[10, 9, 9]])
print multiclass_softmax(a, 0)

a = np.array([[10, -100, -100]])
print multiclass_svm(a, 0)
a = np.array([[10, -100, -100]])
print multiclass_softmax(a, 0)
