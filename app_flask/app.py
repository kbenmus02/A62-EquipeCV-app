# Lancer le serveur : flask run
# Tutoriel : https://blog.miguelgrinberg.com/post/handling-file-uploads-with-flask
# Images : C:\Users\René\Documents\Rene\IA\IA297\Jupyter\420-A62-BB_ProjetSynthese\A62-EquipeCV\cell_images\Parasitized
import os
import os.path
import imghdr

import pandas as pd
import numpy as np
import cv2

import fnmatch  # Permet de filtrer les noms de fichier selon l'extention.
import glob
import utils

import tensorflow as tf
from tensorflow import keras

from flask import Flask, \
    abort, \
    redirect, \
    render_template, \
    request, \
    send_from_directory, \
    url_for

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"]=50 * 1024 #50 Ko
app.config["UPLOAD_EXTENSIONS"] = [".png"]
app.config["UPLOAD_PATH"] = "uploaded_img" # Le répertoire doit exister avant d'être utilisé.

PATH_ROOT = ".."
PATH_NOTEBOOK = PATH_ROOT + "/notebook"
PATH_MODEL = PATH_ROOT + "/model"
MODEL_FILE_NAME = "model_cnn.h5"
IMG_SIZE = 64
IMG_IN_COLOR = 1

@app.route("/")
def index():
    img_file_name_list = fnmatch.filter(os.listdir(app.config['UPLOAD_PATH']), "*.png") #Liste des images dans le répertoire UPLOAD_PATH.
    print("### Index(), av predict()")
    prediction_table_row_list = predict()
    print("###", prediction_table_row_list)
    print("### Index(), ap predict()")
    return render_template('index.html', file_name_list=img_file_name_list, table_row_list = prediction_table_row_list)

@app.route("/", methods=["POST"])
# Charge les fichiers choisis dans le répertoire de destination
def manage_button():
    print("### predict_button = [" + str(request.form.get("predict_button")) + "]")

    if request.form.get("predict_button") != None:
        print("### upload_file")
        # uploaded_file = request.files['img_file'] # Mettre le "name" du champ dans le formulaire HTML

        #=== Efface les fichiers existants
        file_name_list = glob.glob(app.config["UPLOAD_PATH"] + '/*.png')
        for file_name in file_name_list:
            os.remove(file_name)
        #=== Efface les fichiers existants

        #=== Charge les fichiers sélectionnés dans le répertoire [UPLOAD_PATH]
        for uploaded_file in request.files.getlist("img_file"): # Mettre le "name" du champ dans le formulaire HTML
            filename =  secure_filename(uploaded_file.filename) # Élimine les caractères indésirables du nom du fichier.

            if filename != "":
                print("### filename", filename)
                file_ext = os.path.splitext(filename)[1]

                if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                    return "Invalid image", 400
                elif not is_valid_png_img(uploaded_file.stream):
                    return "Invalid image", 400
                else:
                    uploaded_file.save(os.path.join(app.config["UPLOAD_PATH"], filename))
        #--- Charge les fichiers sélectionnés dans le répertoire [Upload]

        #une fois le chargement fait, la prédiction va être faite automatiquement
        return redirect(url_for("index"))

#=== Pour prétraitement des images
def normalize_pixels(img_arr: np.array) -> np.array:
    img_arr_norm = img_arr / 255.0
    return img_arr_norm.astype("float16")

def image_resize(path_img: str) -> np.array:
    img_arr = cv2.imread(path_img, flags=IMG_IN_COLOR)

    h, w, _ = img_arr.shape
    ratio = IMG_SIZE / max(h, w)
    img_arr_resize = cv2.resize(img_arr, dsize=(int(np.ceil(w * ratio)), int(np.ceil(h * ratio))))
    return img_arr_resize

def preprocess_img(path_img: str, img_size: int = IMG_SIZE) -> np.array:
    # img_arr_std = positive_global_std(image_resize(path_img))
    img_arr_preprocessed = normalize_pixels(image_resize(path_img))
    # padding
    img_arr_padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=type(img_arr_preprocessed[0, 0, 0]))
    img_arr_padded[:img_arr_preprocessed.shape[0], :img_arr_preprocessed.shape[1], :] = img_arr_preprocessed

    return img_arr_padded
#--- Pour prétraitement des images

def predict():
    print("### Prédiction")

    """
    #=== Charge les fonctions pour le prétraitement
    print("### Chargement des fonctions de pré-traitement ...")
    import dill
    preprocessing_function_name_list = ["normalize_pixels", "preprocess_img"]

    preprocessing_function_name = preprocessing_function_name_list[0]
    print("### Import de [" + preprocessing_function_name + "]")
    with open(PATH_NOTEBOOK + "/" + preprocessing_function_name + ".dill", "rb") as file:
        normalize_pixels = dill.load(file)
    print(normalize_pixels.__name__)

    preprocessing_function_name = preprocessing_function_name_list[1]
    print("### Import de [" + preprocessing_function_name + "]")
    with open(PATH_NOTEBOOK + "/" + preprocessing_function_name + ".dill", "rb") as file:
        preprocess_img = dill.load(file)
    print(preprocess_img.__name__)
    print("### Chargement des fonctions de pré-traitement fait.")
    #--- Charge les fonctions pour le prétraitement
    """

    # === Charge le modèle
    print("### Chargement du modèle ...")
    model = keras.models.load_model(PATH_MODEL+ "/" + MODEL_FILE_NAME)
    #model = keras.models.load_model(PATH_MODEL+ "/model_cnn.h5")
    #model = utils.pickle_read(PATH_MODEL + "/" + MODEL_FILE_NAME)
    print("### " + str(model))
    print("### Chargement du modèle fait")
    # --- Charge le modèle

    # === Charge la liste des images
    print("### Chargement de la liste des images ...")
    img_file_name_list = glob.glob(app.config["UPLOAD_PATH"] + '/*.png')
    df_img_file_name = pd.DataFrame(img_file_name_list, columns=["img_file_info"])
    df_img_file_name["img_prediction"] = ""
    print(df_img_file_name)
    print("### Chargement de la liste des images fait.")
    # --- Charge de la liste des images

    # === Fait la prédiction sur les images
    print("### Prédiction pour les images ...")

    for i in range(0, len(df_img_file_name)):
        print("###", df_img_file_name.iloc[i]["img_file_info"])
        preprocessed_img_arr = preprocess_img(df_img_file_name.iloc[i]["img_file_info"], IMG_SIZE).reshape((1, 64, 64, 3))
        img_prediction = model.predict(preprocessed_img_arr)
        print("###", img_prediction)
        df_img_file_name.at[i, "img_prediction"] = img_prediction

    print(df_img_file_name)
    print("### Prédiction pour les images faite.")
    # --- Fait la prédiction sur les images
    return df_img_file_name.to_html()

# Vérifie si l'image est bien au format .png
def is_valid_png_img(stream)->bool:
    result=False

    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    print("### format", format)

    if not format:
        result=False
    elif format == "png":
        result=True

    return result

# Retourne la liste des fichiers dans le répertoire
#  Pour l'affichage des images
@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_PATH"], filename)

def load_model(model_file_info: str):
    #=== Charge le modèle
    NOTEBOOK_PATH = "../notebook"
    model = utils.pickle_read(model_file_info)
    return model
    #--- Charge le modèle

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port)