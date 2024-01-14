import cv2
from flask import Flask, request
from flask_restful import Resource, Api

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


class PeopleCounterStaticDynamicUrl(Resource):
    def get(self)
        url = request,args,get('url')
        print ('url', url)
        return {'PeopleCount': 0}

    #todo: pobrac zdjecie z adresu,
    #pobrane zdjecie mozna zapisac na dysku lub przetwarzac je w pamieci podrecznej
    #zaladowane zdjecie do zmiennej image przekazac do algorytmu i zwrocic z endopintu liczbe wykrytych osob
#url = 'https://place.dog/300/200'

>>> # fetch file
>>> #response = requests.get(url, allow_redirects=True)

>>> # Get response status
>>> #response.status_code
#200



api.add_resource(PeopleCounterStatic, '/')

if __name__ == '__main__':
    app.run(debug=True)
