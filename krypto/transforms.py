import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import random
import pandas as pd
import numpy as np
from torchvision.transforms import ToTensor

class FaceCutOut(ImageOnlyTransform):
    
    def __init__(self, landmarks_df="points_df_with_img_index.csv", p=0.5, option="all"):
        super().__init__(p=1)
        self.p = p
        self.landmarks_df = pd.read_csv(landmarks_df).set_index('img_index')
        self.landmarks_df = self.landmarks_df.astype(int)
        self.option = option

    def apply(self, image, img_index, **params):
        rand = random.random()
        if rand < self.p:
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
        if self.option == "all":
            options = ['nose', 'mouth', 'eyes']
            selected_option = random.choice(options)
        elif self.option == "eyes":
            selected_option = 'eyes'
        elif self.option == "nose":
            selected_option = 'nose'
        elif self.option == "mouth":
            selected_option = 'mouth'
        
        if selected_option == 'eyes':
            eyes = ['left_eye_x','left_eye_y','right_eye_x','right_eye_y']
            return self._cutout_eyes(image, landmarks[eyes].to_list())
        elif selected_option == 'nose':
            nose = ['nose_x','nose_y']
            return self._cutout_nose(image, landmarks[nose].to_list())
        elif selected_option == 'mouth':
            mouth = ['left_mouth_x','left_mouth_y','right_mouth_x','right_mouth_y']
            return self._cutout_mouth(image, landmarks[mouth].to_list())

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
        shift_x = 50
        shift_y = 20
        shift = np.array([-shift_x, -shift_y, shift_x, shift_y])
        eyes = np.array(landmarks)
        eyes += shift
        return eyes

    def _get_bbox_from_nose(self, landmarks):
        shift_x = 10
        shift_y = 60
        landmarks = np.concatenate([landmarks, landmarks])
        shift = np.array([-shift_x, -shift_y, shift_x, shift_y])
        nose = np.array(landmarks)
        nose += shift
        return nose

    def _get_bbox_from_mouth(self, landmarks):
        shift_x = 50
        shift_y = 20
        shift = np.array([-shift_x, -shift_y, shift_x, shift_y])
        mouth = np.array(landmarks)
        mouth += shift
        return mouth

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
        return {"image": ["img_index"]}
    


class ToTensorAlbu(ImageOnlyTransform):
    def __init__(self):
        super().__init__(p=1)
        self.to_tensor = ToTensor()

    def apply(self, image, **params):
        return self.to_tensor(image)