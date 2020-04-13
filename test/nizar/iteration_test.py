from logger import logger
from annotation_utils.coco.refactored.structs import COCO_Dataset

dataset = COCO_Dataset.load_from_path(
    json_path='/home/nizar/Desktop/hole/interphone0/coco/output_interphone0_bbox.json',
    img_dir='/home/nizar/Desktop/hole/interphone0/img'
)

print([coco_cat.name for coco_cat in dataset.categories])

# for coco_cat in dataset.categories:
#     print(f'coco_cat:\n{coco_cat}')