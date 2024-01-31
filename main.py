import cv2
from flask import Flask, request
from flask_restful import Resource, Api
from urllib.request import urlretrieve

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

app = Flask(__name__)
api = Api(app)

class PeopleCounterStatic(Resource):

    def get(self):
        # load image
        image = cv2.imread('klasowe-bez-ramki.jpg')
        image = cv2.resize(image, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}
class PeopleCounterDynamicUrl(Resource):
    def get(self):
        url = 'https://img.freepik.com/free-photo/people-surfing-brazil_23-2151079355.jpg?w=996&t=st=1706561159~exp=1706561759~hmac=60f4370805a82af086129a965e83740cc569753d7c75ff6879c9d72070b0204c'
        img_path, _ = urlretrieve(url, "downloaded_image.jpg")
        image2 = cv2.imread(img_path)
        image2 = cv2.resize(image2, (700, 400))

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image2, winStride=(4, 4), padding=(8, 8), scale=1.05)

        return {'peopleCount': len(rects)}


api.add_resource(PeopleCounterStatic, '/')
api.add_resource(PeopleCounterDynamicUrl, '/dynamic')

if __name__ == '__main__':
    app.run(debug=True)
