import os
import argparse
import cv2
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

from utils.align import AlignDlib
from openface_keras.model import create_model

parser = argparse.ArgumentParser()
parser.add_argument('--openface_path', required=True)
parser.add_argument('--dlib_path', required=True)
parser.add_argument('--data_dir', required=True)
parser.add_argument('--output_dir', default='embed_data')

class Preprocess:
    def __init__(self, landmark_path, openface_path):
        self.alignment = AlignDlib(landmark_path)
        self.model = create_model()
        self.model.load_weights(openface_path)

    def auto_align(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        face = self.alignment.getLargestFaceBoundingBox(img)
        face_aligned = self.alignment.align(96, img, face, 
            landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

        return face_aligned
    
    def auto_embed(self, path):
        face_aligned = self.auto_align(path)
        if face_aligned is None:
            return None
        embedding = self.model.predict([np.expand_dims(face_aligned, axis=0)])[0]
        return embedding

    def get_embed(self, path):
        return self.auto_embed(path)


if __name__ == "__main__":
    args = parser.parse_args()
    train_img_dir = os.path.join(args.data_dir, 'train')
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    tool = Preprocess(args.dlib_path, args.openface_path)
    embeds = []
    labels = []
    for i,row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(train_img_dir, row['image'])
        label = row['label']
        em = tool.get_embed(img_path)
        if em is not None:
            embeds.append(em)
            labels.append(label)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    embeds = np.stack(embeds)
    labels = np.array(labels)

    np.save(os.path.join(args.output_dir, 'embed_img.npy'), embeds)
    np.save(os.path.join(args.output_dir, 'label.npy'), labels)



