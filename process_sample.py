import concurrent.futures
import os
import time

import cv2 as cv
import numpy as np
import onnxruntime as rt
import random

from settings import *
from bbox import PredictedObj
from nms import NonMaximumSupression


def process_sample(sample_path):
    '''PROCESSING'''

    sample_name = os.path.basename(sample_path)
    image = cv.imread(sample_path)

    sample = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    sample = np.expand_dims(np.transpose(
        np.array(sample, dtype='float32'), [2, 0, 1]) / 255., axis=0)

    detections = sess.run(output_names, {input_name: sample})
    sample_detections = []

    for b, s in zip(detections[0][0], detections[1][0]):  # ?
        if (max(s) > 0.2):
            class_id = np.argmax(s)
            confidence = s[class_id]

            x1, y1, x2, y2 = tuple([int(i * 832) for i in b[0]])

            sample_detections.append(PredictedObj(class_id,
                                                  x1, y1, x2, y2, confidence))

    # sample_detections = NonMaximumSupression(sample_detections, nms_threshold)

    for bbox in sample_detections:
        cv.rectangle(image, bbox.get_bbox_for_CV2rectangle(),
                     object_colors[bbox.get_label()], 1)

    cv.imwrite('withoutNMS.jpg', image)
    # cv.imshow('img', image)
    # cv.waitKey()


if __name__ == '__main__':
    sess = rt.InferenceSession(model)

    outputs = sess.get_outputs()  # ?
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name

    total_start = time.time()
    images = [1]

    process_sample('176.JPG')

    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # ?
    #     executor.map(process_sample, images[:len(images) // 13])

    total_finish = time.time()

    print(f'Total processing time: {(total_finish - total_start):.2f} sec',
          f'Average processing time: {(total_finish - total_start) / len(images):.2f} sec/img',
          f'Processing speed: {len(images) / (total_finish - total_start):.2f} img/sec',
          sep='\n')
