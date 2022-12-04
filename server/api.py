import base64
import zlib
import time
import torch
import numpy as np
import src.dataloder

from PIL import Image
from flask import Flask, jsonify, render_template, request, make_response, flash, redirect, send_file
from cassandra_storage import create_connection, insert_data, create_database, get_specific_data, get_images, delete_specific_image
from src.model import Net
from torchvision import transforms, utils


app = Flask(__name__)
time.sleep(75)
cursor = create_connection()
ALLOWED_EXTENSIONS = {'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def p():
    resp = jsonify({'message': 'HELLO'})
    resp.status_code = 201
    return resp


@app.route('/server/auth', methods=['POST'])
def login_method():
    resp = jsonify({})
    resp.status_code = 201
    return resp


@app.route('/server/images', methods=['GET'])
def get_images_server():
    resp = jsonify({'data': get_images(cursor)})
    resp.status_code = 200
    return resp


@app.route('/server/images', methods=['POST'])
def upload_images():
    if 'file' not in request.files:
        return make_response('No file found', 400)
    file = request.files['file']
    if file.filename == '':
        return make_response('No image selected for uploading', 404)
    if file and allowed_file(file.filename):
        file.save('tmp.jpeg')
        image = open('tmp.jpeg', 'rb')  # open binary file in read mode
        image_read = image.read()
        image_64_encode = base64.encodebytes(image_read)
        compressed = zlib.compress(image_64_encode)
        insert_data(cursor, compressed, file.filename)
        return make_response("Ok! Image successfully uploaded", 200)
    else:
        return make_response('Allowed image types are -> jpeg', 400)


@app.route('/server/images/<id>', methods=['GET'])
def get_specific_image(id):
    encoded_data = get_specific_data(cursor, id)
    decompressed = zlib.decompress(encoded_data)
    decoded = base64.decodebytes(decompressed)
    picture = open('image.jpeg', 'wb')
    picture.write(decoded)
    return send_file("image.jpeg", as_attachment=True)


@app.route('/server/images/<id>', methods=['DELETE'])
def delete_image(id):
    delete_specific_image(cursor, id)
    resp = jsonify({})
    resp.status_code = 201
    return resp


@app.route('/server/ml_model/<id>', methods=['GET'])
def upload_model(id):
    classes = ['BENIGN', 'MALIGNANT']
    n_classes = 2
    encoded_data = get_specific_data(cursor, id)
    decompressed = zlib.decompress(encoded_data)
    decoded = base64.decodebytes(decompressed)
    picture = open('image.jpeg', 'wb')
    picture.write(decoded)

    # load the model
    model = Net()
    model.load_state_dict(torch.load('/app/model_adam_0001_cpu.pth'))

    img = Image.open('image.jpeg')
    # transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))])
    img = transform(img)
    img = np.array(img)
    img = img[np.newaxis, :, :, :]
    # convert to tensor
    img = torch.from_numpy(img)

    # get the prediction
    pred = model(img)
    # get the index of the max log-probability
    pred = pred.argmax(dim=1, keepdim=True).numpy()[0][0]
    if pred == 0:
        output_msg = f"The model predicted that the tumor is {classes[pred]}"
    else:
        output_msg = f"Madam {id.split('.')[0].replace('_', ' ')} should be contacted as soon as possible and appointment should be prioritzied"
    resp = jsonify({'result': output_msg})
    resp.status_code = 201
    return resp


@app.route('/server/ml_model', methods=['UPDATE'])
def update_model():
    resp = jsonify({})
    resp.status_code = 201
    return resp


if __name__ == "__main__":
    create_database(cursor)
    app.run(host='0.0.0.0')
