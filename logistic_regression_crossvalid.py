import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score

creditData = pd.read_csv(r"C:\Users\JawadSajid\Desktop\Learning\Course 2\Datasets\Datasets\credit_data.csv")
features = creditData[["income","age","loan"]]
target = creditData.default

model = LogisticRegression()
predicted = model_selection.cross_val_predict(model,features,target, cv=3)

print(accuracy_score(target,predicted))