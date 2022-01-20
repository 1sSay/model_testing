# For video testing
video_name = 'DJI_0525.MP4'
predicted_video_name = 'predicted_video.avi'

# Predicted
predicted_images = 'predicts\\'
true_bboxes = 'probnii\\'

IoU_threshold_TP = 0.75
IoU_threshold_FP = 0.10

# Frame
blob_size = 832
overlapping = 0.5

model = 'yolov4_1_3_832_832_static_simp.onnx'

# Thresholds
confidence_threshold = 0.6
nms_threshold = 0.3
width_and_height_threshold = 30
area_threshold = {
    0: 1200,
    1: 1600,
    2: 1600
}

# Objects
classes = {
    0: "car",
    1: "bus",
    2: "truck"
}
object_colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0)
}
