# import requirements needed
from flask import Flask, render_template
from utils import get_base_url

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
@app.route(f'{base_url}')
def home():
    return render_template('index.html')

# Link to Homepage
@app.route(f'{base_url}/homelink')
def returnHome():
    return render_template('index.html')

# Link to EDA
@app.route(f'{base_url}/edalink')
def returnEDA():
    return render_template('eda.html')

# Link to Modelling
@app.route(f'{base_url}/modellink')
def returnModeling():
    return render_template('modelling.html')

# Link to Conclusion
@app.route(f'{base_url}/conclusionlink')
def returnConclusion():
    return render_template('conclusion.html')

# Link to About Us
@app.route(f'{base_url}/about_uslink')
def returnAboutUs():
    return render_template('about-us.html')

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc8.ai-camp.dev'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
