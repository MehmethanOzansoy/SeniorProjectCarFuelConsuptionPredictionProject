    # %%
from flask import Flask,render_template,request
import pandas as pd
import pickle



# %%
app = Flask(__name__)
model = pickle.load(open("new_predict.pkl",'rb'))
car = pd.read_csv("new_data.csv")
# %%
@app.route('/')
def index():
    cylinders = sorted(car['cylinders'].unique())
    displacement = sorted(car['displacement'].unique())
    horsepower = sorted(car['horsepower'].unique())
    weight = sorted(car['weight'].unique())
    acceleration = sorted(car['acceleration'].unique())
    model_year = sorted(car['model_year'].unique())
    origin = sorted(car['origin'].unique())
    brand = sorted(car['brand'].unique())
    return render_template('home.html',
                           cylinders=cylinders,displacement=displacement,horsepower=horsepower,
                           weight=weight,acceleration=acceleration,model_year=model_year,origin=origin,brand=brand)


# %%
@app.route('/predict',methods=['POST'])
def predict():
    cylinders = int(request.form.get('cylinders'))
    displacement = int(request.form.get('displacement'))
    horsepower = int(request.form.get('horsepower'))
    weight = int(request.form.get('weight'))
    acceleration = int(request.form.get('acceleration'))
    model_year = request.form.get('model_year')
    origin = request.form.get('origin') 
    brand = request.form.get('brand')
    
    
    prediction = model.predict(pd.DataFrame([[cylinders,displacement,horsepower,weight,acceleration,model_year,origin,brand]],
                                            columns=['cylinders','displacement','horsepower','weight','acceleration','model_year','origin','brand']))

    return str(prediction[0])

# %%
if __name__=='__main__':
    app.run(debug=False)