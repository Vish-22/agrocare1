from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import io
from base64 import encodebytes
from PIL import Image
import os.path
import csv


def predict_crop(district_name, min_temp, max_temp, season):
    def loadData(file):
        saveFile = open(file, "rb")
        obj = pickle.load(saveFile)
        saveFile.close()

        return obj

    def encode(district_name):
        global code

        if district_name in district_name_to_code_map:
            return district_name_to_code_map[district_name]
        else:
            code = code + 1
            district_name_to_code_map[district_name] = code
            return code

    district_name_to_code_map = loadData("district_names_map")
    data_range = loadData("data_range")
    data_min = loadData("data_min")
    classifier = loadData("knn_classifier")
    crops = loadData("crop_labels")

    temp_min = data_min[1]
    rainfall_min = data_min[2]

    district_data_size = len(district_name_to_code_map) - 1
    temp_data_size = data_range[1]
    rainfall_data_size = data_range[2]

    # main logic flow starts here
    rainfall_map_file = pd.read_csv("rain.csv")

    test_district = district_name
    test_temperature = (float(min_temp) + float(max_temp)) / 2
    test_rainfall = (
        float(
            rainfall_map_file.loc[rainfall_map_file["districts"] == test_district].iat[
                0, 1
            ]
        )
        / 10
    )

    to_find_district_normalized = (encode(test_district) - 1) / district_data_size
    to_find_temp_normalized = (test_temperature - temp_min) / temp_data_size
    to_find_rainfall_normalized = (test_rainfall - rainfall_min) / rainfall_data_size

    prediction = classifier.predict(
        [
            [
                to_find_district_normalized,
                to_find_temp_normalized,
                to_find_rainfall_normalized,
            ]
        ]
    )[0]

    crops_list = crops.unique()
    predictions = classifier.predict_proba(
        [
            [
                to_find_district_normalized,
                to_find_temp_normalized,
                to_find_rainfall_normalized,
            ]
        ]
    )[0]

    df = pd.DataFrame(
        [[prediction, test_district]],
        columns=["prediction", "district"],
    )
    df2 = pd.DataFrame(columns=["crop", "probability"])

    print("Predicetd Crop is :", prediction)

    for i in range(len(crops_list)):
        df2.loc[i] = [crops_list[i], predictions[i]]

    df2.to_csv("output2.csv")
    df.to_csv("output.csv")
    def get_response_image(img_path):
        pil_img = Image.open(img_path, mode = 'r')
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG')
        encoded_img =  encodebytes(byte_arr.getvalue()).decode('ascii')
        return encoded_img
    image_path=''
    desc=''
    ph=''
    
    with open('database.csv') as f:
        reader=csv.reader(f)
        for row in reader:
            if(row[0]==prediction):
                desc=row[1]
                image_path=row[3]
                ph=row[2]
    encoded_img = get_response_image(image_path)         
    data={}
    data['crop']=prediction
    data['desc']=desc
    data['ph']=ph
    data['img']= encoded_img
    return data

code = 0
