from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression



X, y = load_iris(return_X_y=True)
print('>>>>>>>>', y)
# print('>>>>>>>>',X[0:4])

# print('>>>>>>>>',X[:2])
print('>>>>>>>>',X[:3, 0:2])
# Rows 1 to 3
# Columns 0 to 1


clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', C=1.0,max_iter=1000).fit(X, y)
print(clf.predict(X[:, :]))
# print(clf.predict_proba(X[:2, :]))
# print(clf.score(X, y))