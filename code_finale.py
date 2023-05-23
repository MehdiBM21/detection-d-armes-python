import object_detection
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

#infos sur le modèle qu'on va utiliser 
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
#chemins:
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Construire un modele depuis les donnees du pipeline
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restaurer le dernier checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-7')).expect_partial()

@tf.function
def detect_fn(image):#fonction recevant une image sous forme de tenseurs et execute la detection dans cette image
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

import cv2 
import numpy as np
import pygame
from matplotlib import pyplot as plt

#%matplotlib inline

#creer un category index depuis la labelmap
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

def trainedmodel(IMAGE_P):#fonction pour detection d'images
    img = cv2.imread(IMAGE_P)
    image_np = np.array(img)#convertir la video image par image en tableau numpy
    #convertir l'image en tenseurs compatible avec tensorflow
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)#appel de la fonction de detection sur l'image convertie

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()#resultat est l'image en format numpy avec les detections effectuees
        #tracer la boite englobante
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)
    

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.savefig("static/mygraph.png")#sauvegarder l'image resultat dans static/mygraph.png


from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_image():
    #recuperer image uploade et la mettre dans static/image.png
    image = request.files['photo']
    image.save('static/image.png')#recuperer image uploade et la mettre dans static/image.png
    return redirect(url_for('show_image'))

@app.route("/step2")
def show_image():#afficher l'image resultat
    image_pat = 'static/image.png'
    trainedmodel(image_pat)#appel de la fonction de detection d'images
    return render_template("show_image.html", image_path='mygraph.png')


@app.route("/camera")
def camera():#fonction pour detection en temps reel
    cap = cv2.VideoCapture(0)#ouvrir camera et recuperer ses donnees
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pygame.mixer.init()
    detection_sound = pygame.mixer.Sound('alarm.wav')


    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)#convertir la video image par image en tableau numpy

        #convertir l'image en tenseurs compatible avec tensorflow
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)#appel de la fonction de detection sur l'image convertie
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy() #resultat est l'image en format numpy avec les detections effectuees
        #tracer la boite englobante
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.75,
                    agnostic_mode=False)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        if detections['detection_scores'][0] >= 0.75: # Si il y a detection, emmetre le son d'alarme
            detection_sound.play()
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    return render_template("index.html")
@app.route("/video", methods=['POST'])
def video():#fonction pour detection dans une video
    video = request.files['video']
    video.save('static/video.mp4')#uploader la video dans le chemin static/video.mp4
    cap = cv2.VideoCapture('static/video.mp4')#recuperer la video avec opencv
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pygame.mixer.init()
    detection_sound = pygame.mixer.Sound('alarm.wav')

    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)#convertir l'image en tableau numpy
        #convertir l'image en tenseurs compatible avec tensorflow
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)#appel de la fonction de detection sur l'image convertie
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy() #resultat est l'image en format numpy avec les detections effectuees
        #tracer la boite englobante
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.75,
                    agnostic_mode=False)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

        if detections['detection_scores'][0] >= 0.75:  # Si il y a detection, emmetre le son d'alarme
            detection_sound.play()
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)







