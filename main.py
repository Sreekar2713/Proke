from fastapi import Request, FastAPI
from starlette.responses import FileResponse 
from pydantic import BaseModel
app = FastAPI()


import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report, roc_curve,precision_recall_curve, auc,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.impute import KNNImputer



df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()
labels =df['stroke'].value_counts(sort = True).index
sizes = df['stroke'].value_counts(sort = True)

le = LabelEncoder()
en_df = df.apply(le.fit_transform)
en_df.head()

df = df.drop('id', axis=1)
len_data = len(df)
len_w = len(df[df["gender"]=="Male"])
len_m = len_data - len_w

men_stroke = len(df.loc[(df["stroke"]==1)&(df['gender']=="Male")])
men_no_stroke = len_m - men_stroke

women_stroke = len(df.loc[(df["stroke"]==1) & (df['gender']=="Female")])
women_no_stroke = len_w - women_stroke

labels = ['Men with stroke','Men healthy','Women with stroke','Women healthy']
values = [men_stroke, men_no_stroke, women_stroke, women_no_stroke]

features=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type',
       'smoking_status']
from matplotlib.offsetbox import AnchoredText
correlation_table = []
for cols in features:
    y = en_df["stroke"]
    x = en_df[cols]
    corr = np.corrcoef(x, y)[1][0]
    dict ={
        'Features': cols,
        'Correlation coefficient' : corr,
        'Feat_type': 'numerical'
    }
    correlation_table.append(dict)
dF1 = pd.DataFrame(correlation_table)

from sklearn.ensemble import ExtraTreesClassifier

X = en_df[features]
y = en_df['stroke']
en_df_imputed = en_df
imputer = KNNImputer(n_neighbors=4, weights="uniform")
imputer.fit_transform(en_df_imputed)

from imblearn.over_sampling import SMOTE
X , y = en_df_imputed[features],en_df_imputed["stroke"]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=23)
sm = SMOTE()
X_res, y_res = sm.fit_resample(x_train,y_train)


data = {'gender': 0 ,
        'age': 88,'hypertension':0,'heart_disease':1,'ever_married': 1,'work_type':2,'Residence_type':1,'smoking_status':1  } 
new = pd.DataFrame.from_dict([data])
  


    #x_train,x_test,y_train,y_test = train_test_split(features,labels, test_size=0.2, random_state=23)
model = KNeighborsClassifier()
model.fit(x_train,y_train)

    







@app.get("/")
async def read_items():
    return FileResponse('index.html')

@app.post("/predict")
async def getdata(request: Request):
    try:
        print("here")
        body = await request.json()
        for k,v in body.items():
            body[k] = int(v)
      
        pdf = pd.DataFrame.from_dict([body])
        print(pdf)
        print(model.predict(pdf))
        return str(model.predict(pdf))
    except Exception as e:
        print(e)

