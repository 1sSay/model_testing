import concurrent.futures
import os
import time
import glob
import cv2 as cv
import numpy as np
import onnxruntime as rt
import math
from sys import stdout, stdin

from settings import *
from bbox import PredictedObj
from nms import NonMaximumSupression



def process_sample(image, x, y):
    sample = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    sample = np.expand_dims(np.transpose(np.array(sample, dtype='float32'), [2, 0, 1]) / 255., axis=0)

    detections = sess.run(output_names, {input_name: sample})  # Здесь нейросеть находит объекты
    sample_detections = []

    for b, s in zip(detections[0][0], detections[1][0]):  # обрабатываем output нейросети
        if max(s) > 0.2:
            class_id = np.argmax(s)
            confidence = s[class_id]

            x1, y1, x2, y2 = tuple([int(i * blob_size) for i in b[0]])

            on_border = x1 < 4 or x2 > blob_size - 4 or y1 < 4 or y2 > blob_size - 4

            sample_detections.append(PredictedObj(class_id,
                                                  x1 + x, y1 + y, x2 + x, y2 + y, confidence, on_border))

    sample_detections = NonMaximumSupression(sample_detections, nms_threshold)

    return sample_detections  # возвращаем найденные объекты


if __name__ == '__main__':
    # process model
    sess = rt.InferenceSession(model)

    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name

    total_start = time.time()  # Для вычисления времени работы

    # read video
    cap = cv.VideoCapture(video_name)

    img_width = 3840  # int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    img_height = 2160  # int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    counter = 0  # frame counter

    x_count = math.ceil((img_width - blob_size) / (blob_size * overlapping))
    y_count = math.ceil((img_height - blob_size) / (blob_size * overlapping))

    x_coord = [i * (img_width - blob_size) // x_count for i in range(x_count)] + [img_width - blob_size]
    y_coord = [i * (img_height - blob_size) // y_count for i in range(y_count)] + [img_height - blob_size]

    # if cap.isOpened():
    #     stdout.write(f"\n{video_name} found\n")

    out = cv.VideoWriter(predicted_video_name, cv.VideoWriter_fourcc(*'MJPG'), 20.0, (img_width, img_height))
    
    ret = True
    while cap.isOpened() and ret:
        ret, frame = cap.read()
        if ret and counter % 3 == 0:

    # for image_path in glob.glob(images_folder + '*'):
    #     if image_path.endswith(('.jpg', '.JPG')):
    #         frame = cv.imread(image_path)

            sliced = []  # Разрезаем изображения
            bboxes = []
            for x in x_coord:
                for y in y_coord:
                    sliced.append((frame[y:y + blob_size, x:x + blob_size, :], x, y))

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=4) as executor:  # Распараллеливаем, чтобы работало быстрее
                for cur_bboxes in executor.map(lambda x: process_sample(*x), sliced):
                    for bbox in cur_bboxes:
                        bboxes.append(bbox)

            bboxes = NonMaximumSupression(bboxes,
                                          nms_threshold,
                                          destroy_objects_on_border=True,
                                          destroy_low_confidence_objects=True)

            # with open('predicts\\' + os.path.basename(image_path)[:-4] + '.txt', 'w') as txt_predicts:  #
            for bbox in bboxes:
                cv.rectangle(frame,
                                bbox.get_bbox_for_CV2rectangle(),
                                object_colors[bbox.get_label()],
                                1)
                cv.putText(frame, f'{classes[bbox.get_label()]} {round(bbox.get_confidence() * 100)}%',
                            (bbox.x_min - 10, bbox.y_min - 10),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            object_colors[bbox.get_label()],
                            2)

            #         txt_predicts.write(bbox.get_bbox_for_YOLO())

            cv.imwrite(f'predicted images\\{counter}.JPG', frame)
            # cv.imshow('1', frame)
            # cv.waitKey()
            out.write(frame)

            stdout.write(f'\r{counter} frames have been predicted')
        counter += 1

    cap.release()
    out.release()

    total_finish = time.time()

    print('',  # Вывод основной информации
          f'Total processing time: {(total_finish - total_start):.2f} sec',
          f'Frames found: {counter}',
          f'Average processing time: {(total_finish - total_start) / counter:.2f} sec/img',
          f'Processing speed: {counter / (total_finish - total_start):.2f} img/sec',
          sep='\n')
