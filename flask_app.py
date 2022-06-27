from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from flask_cors import CORS, cross_origin

# reading the pickle file
mlModel = joblib.load('./static/regr.pkl')

app = Flask(__name__)

# disabling cors LOL
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
def search():
  args = request.args
  age = args.get('Age') 
  weight = args.get('Weight')
  
  print(age, weight)

  if(age and weight):
    # make the prediction

    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    prediction = mlModel.predict(x)[0]

    return render_template("index.html", bpVal=str(prediction))
  else:
    return render_template("index.html", bpVal="")

if __name__ == "__main__":
  app.run()
