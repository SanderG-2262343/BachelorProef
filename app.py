from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import re
from main import getCorrelatedPublicationData
import pandas as pd

app = Flask(__name__)

projects_df = pd.read_csv("data/csvs/data_projects_2024_5_FRIS_2.csv")[["projId", "title"]]

# Shows the homepage
@app.route("/", methods=["GET"])
def homepage():
    return render_template("homepage.html")


# Helper function for searching projects for suggestions
@app.route('/search')
def search():
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])

    matches = projects_df[projects_df['title'].str.lower().str.contains(query)].head(10)

    return jsonify(matches.to_dict(orient='records'))


# Reloads the page with the correlated data
@app.route("/correlated", methods=["POST"])
def correlated():
    projId = request.json.get("projId")
    if projId is None:
        return jsonify({"error": "projId is required"})
    k = int(request.json.get("k", 5))
    correlated_data = getCorrelatedPublicationData(projId, k)
    correlated_data = [re.sub(r'\s*Abstract:\s*', r'\nAbstract: ', data, count=1) for data in correlated_data]
    return render_template("results.html", correlated_data=correlated_data)