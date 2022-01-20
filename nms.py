from settings import *


def NonMaximumSupression(objects,
                         nms_threshold=0.4,
                         destroy_objects_on_border=False,
                         destroy_low_confidence_objects=False):
    objects.sort(key=lambda obj: obj.confidence, reverse=True)

    keep_list = list()
    drop_list = set()

    for i in range(len(objects)):
        if i in drop_list:
            continue

        if destroy_low_confidence_objects and objects[i].get_confidence() < confidence_threshold:
            drop_list.add(i)
            continue

        if objects[i].is_on_border and \
                (objects[i].get_width() < width_and_height_threshold or \
                 objects[i].get_height() < width_and_height_threshold or \
                 objects[i].get_area() < area_threshold[objects[i].get_label()]):
            drop_list.add(i)
            continue

        for j in range(i + 1, len(objects)):
            if j in drop_list:
                continue

            IoU = objects[i].get_IoU(objects[j])
            if IoU > nms_threshold and objects[i].get_confidence() > objects[j].get_confidence():
                drop_list.add(j)
                continue
            if IoU > nms_threshold and objects[i].get_confidence() <= objects[j].get_confidence():
                drop_list.add(i)
                break

        if i not in drop_list:
            keep_list.append(objects[i])

    return keep_list
