#Importing required libraries and modules 
import os
import pandas as pd
from flask import Flask, flash, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from mne.io import read_raw_edf
from tqdm import tqdm
import numpy as np
import torch
from config import Config
import matplotlib.pyplot as plt


UPLOAD_FOLDER = 'user_uploads'
ALLOWED_EXTENSIONS = 'edf'
cwd = os.getcwd()
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#resets the application for each session
predictions_file = os.path.join(cwd , "predictions.csv")
hypnogram_file = os.path.join(cwd , "static\\Hypnogram.png")
uploaded_file = os.path.join(cwd , "user_uploads")
try:
    os.remove(predictions_file)
except:
    pass
try:
    os.remove(hypnogram_file)
except:
    pass
for each in os.listdir(uploaded_file):
    file_path = os.path.join(uploaded_file,each)
    os.remove(file_path)


#Allowed file Extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#Landing Page (rendering home.html)
@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # print (filename)
            return render_template("second.html")
        else:
            something = "please upload a valid edf file"
            return render_template('home.html', text = something)
    return render_template('home.html')


# Buffer Page (Page with Predict Button)
@app.route('/main', methods=['GET', 'POST'])
def predictions_page():

    each_prediction_size = 256
    # create the training model
    config = Config()
    script_module_path = config.script_pth
    model = torch.jit.load(script_module_path, map_location=config.device)

    if request.form['Predict'] == 'PREDICT':
        file_name = os.listdir("user_uploads")[0]
        print (file_name)
        def predict(model, edf_path, device, ann_file=None, channel="EEG Fpz-Cz"):
            event_mapping = {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage 4": 3,
                "Sleep stage R": 4,
                "Sleep stage ?": 0,
                "Movement time": 0,
            }
            EPOCH_LEN = 30  # secs
            raw = read_raw_edf(edf_path, preload=True, stim_channel=None)
            sampling_rate = raw.info["sfreq"]

            raw_ch_df = raw.to_data_frame()[[channel]]
            raw_ch_df.set_index(np.arange(len(raw_ch_df)))
            remove_idx = int(raw_ch_df.shape[0] % (EPOCH_LEN * sampling_rate))
            raw_ch_df = raw_ch_df[: raw_ch_df.shape[0] - remove_idx]
            assert raw_ch_df.shape[0] % (EPOCH_LEN * sampling_rate) == 0

            x_dat = torch.tensor(np.array(raw_ch_df).reshape(-1, 1, 3000))
            x_dat = x_dat.to(device).float()
            y_preds = []

            # get how many batches does this edf file has
            no_of_blocks = x_dat.shape[0]//each_prediction_size

            # are there any remaining blocks
            remaining_blocks = x_dat.shape[0]%each_prediction_size

            for i in tqdm(range(no_of_blocks)):

                ret = model(x_dat[i*each_prediction_size:(i+1)*each_prediction_size]).detach().cpu().numpy()
                ret = ret.argmax(axis=1)
                y_preds+=ret.tolist()

            # if there are any remaining blocks then get the predictions for them
            if remaining_blocks != 0 :
                ret = model(x_dat[(i+1)*each_prediction_size:]).detach().cpu().numpy()
                ret = ret.argmax(axis=1)
                y_preds+=ret.tolist()

            return np.array(y_preds)

        edf_path = os.path.join(cwd , 'user_uploads' , str(file_name))
        model.eval()
        with torch.no_grad():
            outs = predict(model, edf_path, config.device)
        outs.tofile('predictions.csv', sep = ',')

        #plotting the Hypnogram
        df = pd.read_csv("predictions.csv", header=None).T
        t = np.arange(len(df))*30/3600
        predict = df.to_numpy()
        predicted = predict.squeeze()
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, predicted, alpha=0.7)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'R'])
        ax.set_xlabel('Time (h)')
        ax.set_title('Hypnogram')
        ax.legend()
        plt.savefig('static/Hypnogram.png', dpi=600, bbox_inches='tight', papertype="a4")
    return render_template('third.html')

# Prediction Page (Page with Hypnogram and Download Links) 
@app.route("/downloads", methods = ['GET','POST'])
def Buttons():

    #link for downloading CSV file
    if request.form.get("download"):
        path = 'predictions.csv'
        return send_file(path, as_attachment = True)

    # link for downloading Hypnogram
    if request.form.get('hypnogram'):
        path1 = 'static\Hypnogram.png'
        return send_file(path1, as_attachment = True)

    if request.form.get("Reset"):

        predictions_file = os.path.join(cwd , "predictions.csv")
        hypnogram_file = os.path.join(cwd , "static\\Hypnogram.png")
        uploaded_file = os.path.join(cwd , "user_uploads")

        try:
            os.remove(predictions_file)
        except:
            pass

        try:
            os.remove(hypnogram_file)
        except:
            pass

        for each in os.listdir(uploaded_file):
            file_path = os.path.join(uploaded_file,each)
            os.remove(file_path)

        return render_template('home.html')

    return render_template('third.html')
    


# @app.route ("/reset", methods = ["POST"])
# def delete():
#     location = cwd
#     if request.form['reset_button'] == 'Reset':
#         predictions = cwd + "predictions.csv"
#         hypnogram = cwd + "static/Hypnogram.py"
#         uploaded_file = cwd + "user_uploads/"
#         os.remove(predictions)
#         os.remove(hypnogram)
#         for each in os.listdir(uploaded_file):
#             file_path = os.path.join(uploaded_file,each)
#             os.remove(file_path)
#         string = 'Reset Done Succesfully'
#     return render_template('home.html', text1 = string)

if __name__=="__main__":
    app.run(debug=False)