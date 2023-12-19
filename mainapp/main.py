from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle

#インスタンス化
app = FastAPI()

#入力するデータの型ヒントを設定
class iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

#学習済みモデルの読み込み
f = open('models/model_iris', 'rb')
model = pickle.load(f)

#トップページ
@app.get('/')
def index():
    return {"iris": "iris_prediction"}

#POSTが送信された時(入力)と予測値(出力)の定義
@app.post('/predict')
def make_predictions(features: iris):
    return({"prediction": str(model.predict([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])[0])})