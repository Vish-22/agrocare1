from flask import Flask, request
from train import train
from prediction import predict_crop

app = Flask(__name__)


@app.route("/predictCrop", methods=["POST", "GET"])
def check():
    if request.method == "POST":
        district = request.form["district"]
        min_temp = request.form["min_temp"]
        max_temp = request.form["max_temp"]
        season = request.form["season"]
        if district == None or min_temp == None or max_temp == None or season == None:
            return "You are sending empty Data"
        train()
        return predict_crop(district, min_temp, max_temp, season)


if __name__ == "__main__":
    app.run()
