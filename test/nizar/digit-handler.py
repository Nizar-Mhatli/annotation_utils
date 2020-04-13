from __future__ import annotations
from typing import List
import json
import cv2
from tqdm import tqdm

from common_utils.check_utils import check_file_exists, check_required_keys, check_dir_exists, check_file_exists
from common_utils.common_types.bbox import BBox
from annotation_utils.base.structs import BaseStructObject, BaseStructHandler

from annotation_utils.coco.refactored.structs import COCO_Dataset, \
    COCO_Info, COCO_License_Handler, COCO_Image_Handler, COCO_Annotation_Handler, COCO_Category_Handler, \
    COCO_Image, COCO_License, COCO_Annotation, COCO_Category

class DigitBox(BaseStructObject['DigitBox']):
    def __init__(self, bbox: BBox, label: str):
        super().__init__()
        self.bbox = bbox
        self.label = label
    
    def __str__(self) -> str:
        return f'DigitBox[Label:{self.label}, BBox:{self.bbox}]'

    @classmethod
    def from_dict(cls, item_dict: dict) -> DigitBox:
        check_required_keys(
            item_dict=item_dict,
            required_keys=[
                'top', 'left', 'height', 'width', 'label'
            ]
        )
        ymin = item_dict['top']
        xmin = item_dict['left']
        bbox_h, bbox_w = item_dict['height'], item_dict['width']
        bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmin+bbox_w, ymax=ymin+bbox_h)
        label = item_dict['label']
        return DigitBox(bbox=bbox, label=label)

    @classmethod
    def load_from_path(cls, json_path: str) -> DigitBox:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return DigitBox.from_dict(item_dict=json_dict)

class DigitBoxList(BaseStructHandler['DigitBoxList', 'DigitBox']):
    def __init__(self, digits: List[DigitBox]=None):
        super().__init__(obj_type=DigitBox, obj_list=digits)
        self.digits = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> DigitBoxList:
        return DigitBoxList(
            digits=[DigitBox.from_dict(item_dict) for item_dict in dict_list]
        )

    @classmethod
    def load_from_path(cls, json_path: str, strict: bool=True) -> DigitBoxList:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return DigitBoxList.from_dict_list(json_data)

class DigitGroup(BaseStructObject['DigitGroup']):
    def __init__(self, filename: str, digits: DigitBoxList=None):
        self.filename = filename
        self.digits = digits if digits is not None else DigitBoxList()

    def __str__(self) -> str:
        return f'DigitGroup{str(self.__dict__)}'

    @classmethod
    def from_dict(cls, item_dict: dict) -> DigitGroup:
        check_required_keys(
            item_dict=item_dict,
            required_keys=[
                'boxes', 'filename'
            ]
        )
        boxes = item_dict['boxes']
        filename = item_dict['filename']
        return DigitGroup(filename=filename, digits=DigitBoxList.from_dict_list(boxes))

    @classmethod
    def load_from_path(cls, json_path: str) -> DigitGroup:
        check_file_exists(json_path)
        json_dict = json.load(open(json_path, 'r'))
        return DigitGroup.from_dict(item_dict=json_dict)

class DigitGroupList(BaseStructHandler['DigitGroupList', 'DigitGroup']):
    def __init__(self, digits: List[DigitGroup]=None):
        super().__init__(obj_type=DigitGroup, obj_list=digits)
        self.groups = self.obj_list

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> DigitGroupList:
        return DigitGroupList(
            digits=[DigitGroup.from_dict(item_dict=item_dict) for item_dict in dict_list]
        )

    @classmethod
    def load_from_path(cls, json_path: str, strict: bool=True) -> DigitGroupList:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return DigitGroupList.from_dict_list(json_data)

    def to_coco(self, img_dir: str, show_pbar: bool=True) -> COCO_Dataset:
        check_dir_exists(img_dir)

        # Initialize Handlers
        info = COCO_Info(description='COCO Dataset Converted From Digits Format')
        licenses = COCO_License_Handler()
        images = COCO_Image_Handler()
        annotations = COCO_Annotation_Handler()
        categories = COCO_Category_Handler()

        # Load Up The Handlers
        licenses.append(
            COCO_License(url='N/A', id=0, name='Free License')
        )

        pbar = tqdm(total=len(self), unit='image(s)') if show_pbar else None
        if pbar is not None:
            pbar.set_description('Converting to COCO...')
        for group in self:
            img_filename = group.filename
            img_path = f'{img_dir}/{img_filename}'
            check_file_exists(img_path)
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            image_id = len(images)
            coco_image = COCO_Image.from_img_path(
                img_path=img_path,
                license_id=0,
                image_id=image_id
            )
            images.append(coco_image)
            bbox_list = []
            for digit in group.digits:
                bbox = digit.bbox
                bbox_list.append(bbox)
                
                # Test
                bbox_h, bbox_w = bbox.shape()
                if bbox_h == 0 or bbox_w == 0:
                    logger.error(f'Encountered bbox with zero area.')
                    logger.error(f'bbox: {bbox}')
                    raise Exception

                label = str(int(float(digit.label)))

                if label not in [coco_cat.name for coco_cat in categories]:
                    categories.append(
                        COCO_Category(
                            id=len(categories),
                            supercategory=label,
                            name=label
                        )
                    )
                coco_cat = categories.get_unique_category_from_name(label)
                annotations.append(
                    COCO_Annotation(
                        id=len(annotations),
                        category_id=coco_cat.id,
                        image_id=image_id,
                        bbox=bbox,
                        area=bbox.area()
                    )
                )
            if 'whole_number' not in [coco_cat.name for coco_cat in categories]:
                categories.append(
                    COCO_Category(
                        id=len(categories),
                        supercategory='whole_number',
                        name='whole_number'
                    )
                )
            coco_cat = categories.get_unique_category_from_name('whole_number')
            bbox0_xmin = min([seg_bbox.xmin for seg_bbox in bbox_list])
            bbox0_ymin = min([seg_bbox.ymin for seg_bbox in bbox_list])
            bbox0_xmax = max([seg_bbox.xmax for seg_bbox in bbox_list])
            bbox0_ymax = max([seg_bbox.ymax for seg_bbox in bbox_list])
            result_bbox = BBox(xmin=bbox0_xmin, ymin=bbox0_ymin, xmax=bbox0_xmax, ymax=bbox0_ymax)
            annotations.append(
                COCO_Annotation(
                    id=len(annotations),
                    category_id=coco_cat.id,
                    image_id=image_id,
                    bbox=result_bbox,
                    area=result_bbox.area()
                )
            )

            if pbar is not None:
                pbar.update(1)

        # Construct COCO Dataset
        dataset = COCO_Dataset(
            info=info,
            licenses=licenses,
            images=images,
            annotations=annotations,
            categories=categories
        )
        return dataset

from logger import logger
ann_path = 'data/digitStruct.json'
img_dir = 'data/images'
digit_groups = DigitGroupList.load_from_path(ann_path)
dataset = digit_groups.to_coco(img_dir=img_dir, show_pbar=True)
dataset.save_to_path(save_path='output.json', overwrite=True)

dataset = COCO_Dataset.load_from_path(json_path='output.json')
dataset.display_preview(bbox_label_thickness=1)