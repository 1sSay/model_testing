class Bbox:
    def __init__(self, label, x1, y1, x2, y2):
        self.label = label

        self.x_min = x1
        self.y_min = y1
        self.x_max = x2
        self.y_max = y2

        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.area = self.width * self.height

    def __repr__(self):
        return self.label, \
               self.x_min, self.y_min, self.x_max, self.y_max, \
               self.width, self.height, self.area

    def __str__(self):
        return f'{self.label} / {self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}'

    def get_data(self):
        return (self.label,
                self.x_min, self.y_min, self.x_max, self.y_max,
                self.width, self.height, self.area)

    def get_bbox(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def get_label(self):
        return self.label

    def get_bbox_for_CV2rectangle(self):
        return self.x_min, self.y_min, self.width, self.height

    def get_bbox_for_writing(self):
        return f"{self.label} {self.x_min} {self.y_min} {self.x_max} {self.y_max}\n"
    
    def get_bbox_for_YOLO(self, img_width=3840, img_height=2160):
        x, y = round((self.x_min + self.width / 2) / img_width, 6), round((self.y_min + self.height / 2) / img_height, 6)
        w, h = round(self.width / img_width, 6), round(self.height / img_height, 6)
        return f"{self.label} {x} {y} {w} {h}\n"

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_area(self):
        return self.area

    def get_intersection_area(self, second):
        dx = min(self.x_max, second.x_max) - max(self.x_min, second.x_min)
        dy = min(self.y_max, second.y_max) - max(self.y_min, second.y_min)

        if dx > 0 and dy > 0:
            return dx * dy

        return 0

    def get_union_area(self, second):
        return self.area + second.area - self.get_intersection_area(second)

    def get_IoU(self, second):
        intersection_area = self.get_intersection_area(second)
        return intersection_area / (self.area + second.area - intersection_area)


class PredictedObj(Bbox):
    def __init__(self, label, x1, y1, x2, y2, confidence=1, on_border=False):
        self.confidence = confidence
        self.on_border = on_border

        super(PredictedObj, self).__init__(label, x1, y1, x2, y2)

        self.found = False

    def __repr__(self):
        return self.label, self.confidence, \
               self.x_min, self.y_min, self.x_max, self.y_max, \
               self.width, self.height, self.area

    def __str__(self):
        return f'{self.label} / {self.confidence} / {self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}'

    def get_bbox_for_writing(self):
        return f"{self.label} {self.confidence} {self.x_min} {self.y_min} {self.x_max} {self.y_max}\n"

    def get_data(self):
        return (self.image_name, self.label, self.confidence,
                self.x_min, self.y_min, self.x_max, self.y_max,
                self.width, self.height, self.area)

    def get_confidence(self):
        return self.confidence

    def is_on_border(self):
        return self.on_border

    def set_found(self, val):
        self.found = val


class TrueObj(Bbox):
    def __init__(self, label, x1, y1, x2, y2):
        super(TrueObj, self).__init__(label, x1, y1, x2, y2)

        self.found = False

    def set_found(self, val):
        self.found = val

    def get_bbox_for_writing(self):
        return f"{self.label} {self.x_min} {self.y_min} {self.x_max} {self.y_max}"
