#!/usr/bin/env python3

from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
from time import sleep
from keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="templates")
# static folder
app.static_folder = "static"
# upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXT = ["gz", "nii"]
FILE_NO = 0
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
# secret key
app.secret_key = "jesher123"

model = load_model("imagre.h5")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=("POST", "GET"))
def up():
    if request.method == "POST":
        STAT_MESSAGE = ""
        uploaded_img = request.files["uploaded-file"]
        # Extracting uploaded data file name
        img_filename = secure_filename(uploaded_img.filename)
        global FILE_NO
        FILE_NO = img_filename.split("_")[1]
        STAT_MESSAGE = (
            "fMRI data upload success !"
            if img_filename.split(".")[-1] in ALLOWED_EXT
            else "Wrong File !!!!"
        )
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(
            os.path.join(basedir, app.config["UPLOAD_FOLDER"], img_filename)
        )
        # uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session["uploaded_img_file_path"] = os.path.join(
            app.config["UPLOAD_FOLDER"], img_filename
        )
        return render_template(
            "show_image.html",
            upload_status=STAT_MESSAGE,
            process_stat="Processing Please wait.......",
        )


@app.route("/show_image", methods=("POST", "GET"))
def combine():
    global FILE_NO
    in_file_path = "static/inputs/" + str(FILE_NO) + ".JPEG"
    outs = model.predict(session["uploaded_img_file_path"])
    name = str(FILE_NO) + ".png"
    sleep(3)
    plt.imsave(name, outs)
    out_file_path = "static/outputs/" + name
    return render_template(
        "show_image.html",
        gen_image=out_file_path,
        act_image=in_file_path,
        process_stat="Reconstruction Complete !!!",
    )


if __name__ == "__main__":
    app.run(debug=True)
