import numpy as np
import tensorflow as tf
import cv2
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import time
from PIL import Image
from utils import label_map_util
import glob
from utils import visualization_utils as vis_util
class ObjectDetector(object):
    def __init__(self):
        self.label_map = label_map_util.load_labelmap("label_spot.pbtxt")
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=9,use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        # Load a (frozen) Tensorflow model into memory.
       # print(color.BOLD + color.RED + 'INIT GRAPH' + color.END)
        with tf.gfile.GFile('D:/models-1.13.0/research/object_detection/Model-3-7/frozen_inference_graph.pb', 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')
        self.detection_graph = graph

    
    # def __del__(self):
    #     if self.session is not None:
    #         self.session.close()

    def detect(self, sess,image,filename):
        
     
        # Reshape and transform the image in np array
        (im_width, im_height) = image.size
        image = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

        image_np_expanded = np.expand_dims(image, axis=0)
        # Extract image tensor
        image_tensor = self.detection_graph.get_tensor_by_name('prefix/image_tensor:0')
        # Extract detection boxes
        boxes = self.detection_graph.get_tensor_by_name('prefix/detection_boxes:0')
        # Extract detection scores
        scores = self.detection_graph.get_tensor_by_name('prefix/detection_scores:0')
        # Extract detection classes
        classes = self.detection_graph.get_tensor_by_name('prefix/detection_classes:0')
        # Extract number of detectionsd
        num_detections = self.detection_graph.get_tensor_by_name('prefix/num_detections:0')
        # Actual detection.
        start = time.time()
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        end = time.time()
       
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=1,min_score_thresh=0.5)
        #filenamme=image.filename
        print(str(end-start))
        ##cv2.imshow('object detection', image)
        cv2.imwrite(filename[:-4]+'D'+'.jpg', image)
        cv2.waitKey(0)





def main():
        updater = ObjectDetector()
        with tf.Session(graph=updater.detection_graph, config=config) as sess:
                for filename in glob.glob("D:\\check\\moreData\\*.jpg"): #assuming gif
                    image=Image.open(filename)
                    ###image.show()
                    updater.detect(sess,image,filename)
main()
    # ......
 

