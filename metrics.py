import glob
import os
from settings import *
from bbox import Bbox, PredictedObj, TrueObj
from util import FromYOLO, InitPredicted


def GetConfusions(image_name):
    with open(predicted_images + image_name) as txtfile:
        predicted_objects = [[], [], []]  # 0 - cars, 1 - bus, 2 - truck
        for i in txtfile.readlines():
            obj = PredictedObj(*FromYOLO(i))
            predicted_objects[obj.get_label()].append(obj)

    with open(true_bboxes + image_name) as txtfile:
        true_objects = [[], [], []]
        for i in txtfile.readlines():
            obj = TrueObj(*FromYOLO(i))
            true_objects[obj.get_label()].append(obj)

    res = {  # результаты, которые мы будем возвращать
        # TruePositive, FalsePositive, FalseNegative
        'car': {'tp': 0, 'fp': 0, 'tn': 0},
        'bus': {'tp': 0, 'fp': 0, 'tn': 0},
        'truck': {'tp': 0, 'fp': 0, 'tn': 0}
    }

    for obj_label in range(3):  # сопостовляем bboxы
        tp, fp, tn, found = 0, 0, 0, 0

        for true_obj in true_objects[obj_label]:
            found = 0
            for predicted_obj in predicted_objects[obj_label]:
                IoU = true_obj.get_IoU(predicted_obj)
                if IoU == 0:
                    continue
                if IoU >= IoU_threshold_TP:
                    tp += 1
                    found = 1
                    break
                elif IoU >= IoU_threshold_FP:
                    fp += 1
                    found = 1
                    break
            if not found:
                tn += 1

        res[classes[obj_label]]['tp'] += tp
        res[classes[obj_label]]['fp'] += fp
        res[classes[obj_label]]['tn'] += tn

    return res


def CalculateMetrics():
    img_done = 0
    conf = {
        'car': {'tp': 0, 'fp': 0, 'tn': 0},
        'bus': {'tp': 0, 'fp': 0, 'tn': 0},
        'truck': {'tp': 0, 'fp': 0, 'tn': 0}
    }

    for cur_img in glob.glob(predicted_images + "*.txt"):
        img_name = os.path.basename(cur_img)
        res = GetConfusions(img_name)

        for key_obj in res:
            for key_metric in res[key_obj]:
                conf[key_obj][key_metric] += res[key_obj][key_metric]

        img_done += 1
        print(f'\r{img_done} done', end='')

    metrics = {
        'car': dict(),
        'bus': dict(),
        'truck': dict()
    }

    metrics['car']['precision'] = conf['car']['tp'] / \
        (conf['car']['tp'] + conf['car']['fp'])
    metrics['car']['recall'] = conf['car']['tp'] / \
        (conf['car']['tp'] + conf['car']['tn'])
    metrics['car']['F1'] = 2 * metrics['car']['precision'] * \
        metrics['car']['recall'] / \
        (metrics['car']['precision'] + metrics['car']['recall'])

    metrics['bus']['precision'] = conf['bus']['tp'] / \
        (conf['bus']['tp'] + conf['bus']['fp'])
    metrics['bus']['recall'] = conf['bus']['tp'] / \
        (conf['bus']['tp'] + conf['bus']['tn'])
    metrics['bus']['F1'] = 2 * metrics['bus']['precision'] * \
        metrics['bus']['recall'] / \
        (metrics['bus']['precision'] + metrics['bus']['recall'])

    metrics['truck']['precision'] = conf['truck']['tp'] / \
        (conf['truck']['tp'] + conf['truck']['fp'])
    metrics['truck']['recall'] = conf['truck']['tp'] / \
        (conf['truck']['tp'] + conf['truck']['tn'])
    metrics['truck']['F1'] = 2 * metrics['truck']['precision'] * \
        metrics['truck']['recall'] / \
        (metrics['truck']['precision'] + metrics['truck']['recall'])

    return metrics


def PrintMetrcis(metrics):
    print('\n')
    print('Cars:',
          f'    Recall: {round(metrics["car"]["recall"] * 100, 2)}%',
          f'    Precision: {round(metrics["car"]["precision"] * 100, 2)}%',
          f'    F1-score: {round(metrics["car"]["F1"] * 100, 2)}%', sep='\n')

    print('\nBuses:',
          f'    Recall: {round(metrics["bus"]["recall"] * 100, 2)}%',
          f'    Precision: {round(metrics["bus"]["precision"] * 100, 2)}%',
          f'    F1-score: {round(metrics["bus"]["F1"] * 100, 2)}%', sep='\n')

    print('\nTrucks:',
          f'    Recall: {round(metrics["truck"]["recall"] * 100, 2)}%',
          f'    Precision: {round(metrics["truck"]["precision"] * 100, 2)}%',
          f'    F1-score: {round(metrics["truck"]["F1"] * 100, 2)}%', sep='\n')


if __name__ == '__main__':
    print("Calculating metrics:")

    metrics = CalculateMetrics()

    PrintMetrcis(metrics)
