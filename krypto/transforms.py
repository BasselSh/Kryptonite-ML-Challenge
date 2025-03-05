import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import pandas as pd
import numpy as np

class FaceCutOut(ImageOnlyTransform):
    def __init__(self, landmarks_df="points_df_with_img_index.csv", p=0.5):
        self.p = p
        super().__init__(p=1)
        self.landmarks_df = pd.read_csv(landmarks_df)

    def apply(self, image, img_index, **params):
        if random.random() < self.p:
            return self._cutout(image, img_index)
        return image

    def _cutout(self, image, img_index):
        '''
        landmarks = [index,
                    left_eye_x,left_eye_y,right_eye_x,right_eye_y,
                    nose_x,nose_y,
                    left_mouth_x,left_mouth_y,right_mouth_x,right_mouth_y]
        '''
        landmarks = self.landmarks_df.loc[img_index] 
        options = ['eyes', 'nose', 'mouth']
        selected_option = random.choice(options)
        if selected_option == 'eyes':
            return self._cutout_eyes(image, landmarks[1:5])
        elif selected_option == 'nose':
            return self._cutout_nose(image, landmarks[5:7])
        elif selected_option == 'mouth':
            return self._cutout_mouth(image, landmarks[7:11])

    def _cutout_eyes(self, image, landmarks):
        bbox = self._get_bbox_from_eyes(landmarks)
        return self.cutout_bbox(image, bbox)

    def _cutout_nose(self, image, landmarks):
        bbox = self._get_bbox_from_nose(landmarks)
        return self.cutout_bbox(image, bbox)

    def _cutout_mouth(self, image, landmarks):
        bbox = self._get_bbox_from_mouth(landmarks)
        return self.cutout_bbox(image, bbox)
    
    def _get_bbox_from_eyes(self, landmarks):
        pass

    def _get_bbox_from_nose(self, landmarks):
        pass

    def _get_bbox_from_mouth(self, landmarks):
        pass

    def cutout_bbox(self, image, bbox):
        '''
        Zero out pixels inside the bbox.
        bbox is in the following format: top left, bottom right.
        '''
        x1, y1, x2, y2 = bbox
        image[y1:y2, x1:x2] = 0
        return image
    
    @property
    def target_dependence(self):
        return {"image": "img_index"}