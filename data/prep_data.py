import os
import cv2
import dlib
import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
from tqdm import tqdm
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


class TensoflowMobilNetSSDFaceDector():
    def __init__(self,
                 det_threshold=0.3,
                 model_path='model/ssd/frozen_inference_graph_face.pb'):

        self.det_threshold = det_threshold
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            # config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
            #                         allow_soft_placement=True, device_count={'CPU': 1})
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)

    def detect_face(self, image):

        h, w, c = image.shape

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        filtered_score_index = np.argwhere(
            scores >= self.det_threshold).flatten()
        selected_boxes = boxes[filtered_score_index]

        faces = np.array([[
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        ] for y1, x1, y2, x2 in selected_boxes])

        return faces


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    required=True,
                    choices=['imdb', 'wiki', 'utkface', 'fgnet', 'adience'],
                    help='Dataset name')
parser.add_argument('--num_worker',
                    default=2,
                    type=int,
                    help="num of worker")
args = parser.parse_args()
DATASET = args.dataset
WORKER = args.num_worker
predictor = dlib.shape_predictor('../model/shape_predictor_5_face_landmarks.dat')
face_detector = [TensoflowMobilNetSSDFaceDector(
    det_threshold=0.3,
    model_path='../model/ssd/frozen_inference_graph_face.pb')]*WORKER

global_cnt = -1
flag = 0


def detect(image):
    global global_cnt
    global flag
    while flag:
        pass
    flag = 1
    global_cnt += 1
    idx = global_cnt % WORKER
    flag = 0
    # print(global_cnt)
    return face_detector[idx].detect_face(image)


def align_and_save(path: str):
    """
    Get aligned face and save to disk

    Parameters
    ----------
    path : string
        path to image

    Returns
    -------
    integer
        flag to mark. 1 if success detect face, 0 if fail
    """

    RES_DIR = '{}_aligned'.format(DATASET)
    if os.path.exists(os.path.join(RES_DIR, path)):
        return 1
    flname = os.path.join(DATASET, path)
    image = cv2.imread(flname)
    detector = dlib.get_frontal_face_detector()
    faces = detect(image)
    rects = [dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
             for (left, top, right, bottom) in faces]
    # if detect exactly 1 face, get aligned face
    if len(rects) == 1:
        shape = predictor(image, rects[0])
        result = dlib.get_face_chip(image, shape, padding=0.4, size=140)
        folder = os.path.join(RES_DIR, path.split('/')[0])
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        flname = os.path.join(RES_DIR, path)
        if not os.path.exists(flname):
            cv2.imwrite(flname, result)
        return 1
    return 0


def main():
    args = parser.parse_args()
    DATASET = args.dataset
    WORKER = args.num_worker
    data = pd.read_csv('db/{}.csv'.format(DATASET))
    # detector = dlib.get_frontal_face_detector()

    paths = data['full_path'].values
    print('[PREPROC] Run face alignment...')
    with ThreadPool(processes=WORKER) as p:
        res = []
        max_ = len(paths)
        with tqdm(total=max_) as pbar:
            for i, j in tqdm(enumerate(p.imap(align_and_save, paths))):
                pbar.update()
                res.append(j)
        data['flag'] = res

        # create new db with only successfully detected face
        data = data.loc[data['flag'] == 1, list(data)[:-1]]
        data.to_csv('db/{}_cleaned.csv'.format(DATASET), index=False)


if __name__ == '__main__':
    main()
