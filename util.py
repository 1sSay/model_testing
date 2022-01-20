# Переводим считанную строку из YOLO (по дефолту 4к изображение)
def FromYOLO(yolo, W=3840, H=2160):
    label, x, y, w, h = yolo.split()
    label, x, y, w, h = int(label), float(x), float(y), float(w), float(h)

    w, h = w * W, h * H  # Рассчет ширины и высоты
    x, y = x * W - w / 2, y * H - h / 2  # находим левую верхнюю точку
    # Возвращаем кортеж для инициализации TrueObj
    return label, int(x), int(y), int(x + w), int(y + h)


def InitPredicted(obj):
    obj = obj.split()
    return int(obj[0]), float(obj[1]), int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])


def СalculateCounters(path_to_dataset):
    cars = 0
    buses = 0
    trucks = 0

    sum_of_areas_cars = 0
    sum_of_areas_buses = 0
    sum_of_areas_trucks = 0

    count_of_photos = 0

    for path in glob.glob(path_to_dataset + '\\*.txt'):

        count_of_photos += 1

        img_width = 832
        img_height = 832

        with open(path) as txtfile:
            objects = txtfile.readlines()

            for obj in objects:
                obj_class, x, y, w, h = obj.split()

                if obj_class == '0':
                    cars += 1
                    sum_of_areas_cars += ObjAreaFromYOLO(img_width, img_height,
                                                         float(x), float(y),
                                                         float(w), float(h))
                if obj_class == '1':
                    buses += 1
                    sum_of_areas_buses += ObjAreaFromYOLO(img_width, img_height,
                                                          float(x), float(y),
                                                          float(w), float(h))
                elif obj_class == '2':
                    trucks += 1
                    sum_of_areas_trucks += ObjAreaFromYOLO(img_width, img_height,
                                                           float(x), float(y),
                                                           float(w), float(h))

    print(f"Count of photos: \t{count_of_photos}\n\n"
          f"Number of objects: \t{cars + buses + trucks}\n"
          f"Number of cars: \t{cars}\n"
          f"Number of buses: \t{buses}\n"
          f"Number of trucks: \t{trucks}\n\n"
          f"Average areas:\n"
          f"Cars: \t{round(sum_of_areas_cars / max(1, cars))}\n"
          f"Buses: \t{round(sum_of_areas_buses / max(1, buses))}\n"
          f"Trucks:\t{round(sum_of_areas_trucks / max(1, trucks))}\n")