# import requirements needed
from flask import Flask, render_template, request
from utils import get_base_url
import pandas as pd
import numpy as np
import sklearn

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

# set up the routes and logic for the webserver
@app.route(f'{base_url}/indexlink')
def home():
    return render_template('index.html')

#Link to Homepage
@app.route(f'{base_url}')
def returnHome():
    return render_template('index.html')

# Link to EDA
@app.route(f'{base_url}/edalink')
def returnEDA():
    return render_template('eda.html')

# Link to Modelling
@app.route(f'{base_url}/modellink')
def returnModeling():
    return render_template('modelling.html',response=0)

# Link to Conclusion
@app.route(f'{base_url}/conclusionlink')
def returnConclusion():
    return render_template('conclusion.html')

# Link to About Us
@app.route(f'{base_url}/about_uslink')
def returnAboutUs():
    return render_template('about-us.html')

@app.route(f'{base_url}/modellink', methods=["GET", "POST"])
def getFormRequest():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn import tree

    filename = 'insurance.csv'
    my_data = pd.read_csv(filename)
    my_data.dropna(inplace = True)
    my_data.drop_duplicates(inplace = True)
    my_data = my_data.reset_index(drop=True)

    #string to numerical values
    def change_smoke(val):
        if val == 'no':
            return 0
        elif val == 'yes':
            return 1
    my_data['smoker'] = my_data['smoker'].apply(change_smoke)
    def change_region(val):
        if val == 'southwest':
            return 0
        elif val == 'southeast':
            return 1
        elif val == 'northwest':
            return 2
        elif val == 'northeast':
            return 3
    my_data['region'] = my_data['region'].apply(change_region)

    def change_sex(val):
        if val == 'female':
            return 0
        elif val=='male':
            return 1
    my_data['sex'] = my_data['sex'].apply(change_sex)

    X = my_data.drop('charges',axis=1)
    X = X.to_numpy()
    y = my_data["charges"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Create the model object
    linr = LinearRegression()
    mlpr = MLPRegressor(activation='relu', alpha=0.0008, solver='lbfgs', max_iter=8000, max_fun=30000, random_state=3)

    #Fit (train) the model -- this is where the ML happens!
    
    # poly training
    
    poly = PolynomialFeatures(degree=3)
    X_ = poly.fit_transform(x_train)
    linr.fit(X_,y_train)

    # mlp training
    
    target = my_data["charges"]
    input_columns = my_data.loc[:, my_data.columns != "charges"]
    mlpr.fit(x_train, y_train)
    
    
    
    # Decision Tree
    
    insurance = pd.read_csv('insurance.csv')

    insurance.dropna(inplace = True)
    insurance.drop_duplicates(inplace = True)
    insurance = insurance.reset_index(drop=True)

    length = len(insurance.index)
    
    def change_to_binary(sex):

        if sex == "male":
            return 0
        elif sex == "female":
            return 1

    insurance['sex'] = insurance['sex'].apply(change_to_binary)

    def change_to_binary(smoker):

        if smoker == "yes":
            return 0
        elif smoker == "no":
            return 1
    insurance['smoker'] = insurance['smoker'].apply(change_to_binary)

    def change_to_numerical(region):

        if region == "northeast":
            return 0
        elif region == "southeast":
            return 1
        elif region == "northwest":
            return 2
        elif region == "southwest":
            return 3

    insurance['region'] = insurance['region'].apply(change_to_numerical)

    target_x = insurance["charges"]
    input_columns_y =  insurance.loc[:, insurance.columns != "charges"]

    x_trainn, x_testt, y_trainn, y_testt = train_test_split(input_columns, target, test_size=0.2)

    rgsr_tree = tree.DecisionTreeRegressor(criterion= 'squared_error',max_depth= 5, max_leaf_nodes= 5, min_impurity_decrease= 1.0, min_samples_leaf= 10)
    rgsr_tree = rgsr_tree.fit(x_trainn, y_trainn)

    # figure out how to get form answers
    if request.method == "POST":
        age = 0
        if(request.form.get("age") != ''):
            age = int(request.form.get("age"))
        bmi = 0
        if(request.form.get("bmi") != ''):
            bmi = float(request.form.get("bmi"))

        sex = 1
        if(request.form.get("sex") == "Female"):
            sex = 0

        smoker = 0
        if(request.form.get("smoker") == "on"):
            smoker = 1

        region = 0 # 0=southwest, 1=southeast, 2=northwest, 3=northeast
        if(request.form.get("region") == "Southeast"):
            region = 1
        elif(request.form.get("region") == "Northwest"):
            region = 2
        elif(request.form.get("region") == "Northeast"):
            region = 3

        children = 0
        if(request.form.get("children") != ''):
            children = int(request.form.get("children"))

        data = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}
        user_df = pd.DataFrame(data, index = [0])
        poly_user_df = poly.transform(user_df)

        # predicts
        poly_price = linr.predict(poly_user_df)
        mlp_price = mlpr.predict(user_df)
        return render_template("modelling.html", response=int(poly_price))

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc8.ai-camp.dev'

    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
