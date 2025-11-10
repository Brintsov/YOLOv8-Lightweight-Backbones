from quantization.quantizer import Quantizer
from data.prepate_data import parse_annotations, preprocess_images_and_boxes, download_kaggle_data
from models.shufflenet_v2 import create_0_5_shufflenet_yolov8
from models.csp_darknet import create_xs_csp_darknet_yolov8
from models.ghostnet_v2 import create_0_5_ghostnet_yolov8


shuffle_model = create_0_5_shufflenet_yolov8(2)
shuffle_model.load_weights("best_weights/best_shuffle_yolo.keras")
shuffle_model.trainable = False

ghost_model = create_0_5_ghostnet_yolov8(2)
ghost_model.load_weights("best_weights/best_ghost_yolo.keras")
ghost_model.trainable = False

csp_model = create_xs_csp_darknet_yolov8(2)
csp_model.load_weights("best_weights/best_csp_darknet_yolo.keras")
csp_model.trainable = False


path = download_kaggle_data()
annotations_by_image, cat_mapping, cat_mapping_r = parse_annotations(path)
boxes, images, labels = annotations_by_image['boxes'].tolist(), annotations_by_image['image'].tolist(), annotations_by_image['categories'].tolist()
prepared_images, prepared_boxes, prepared_labels = preprocess_images_and_boxes(images, boxes, labels, f"{path}/images", cat_mapping_r)


q = Quantizer(representation_data=prepared_images)
q.quantize_model(ghost_model, 'ghost.tflite')
q.quantize_model(shuffle_model, 'shuffle.tflite')
q.quantize_model(csp_model, 'csp.tflite')
