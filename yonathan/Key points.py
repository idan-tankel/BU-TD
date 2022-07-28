import albumentations as A

# import cv2
# cv2_BORDER_CONSTANT = cv2.BORDER_CONSTANT
cv2_BORDER_CONSTANT = 0


# Declare an augmentation pipeline

def augment_albumentations(img, seg, aug_data, keypoints=None):
    # seed = aug_data.seed
    rotate_deg = aug_data.rotate_deg
    # color_add_range = aug_data.color_add_range
    # xtrans  = aug_data.xtrans
    # ytrans  = aug_data.ytrans
    xtrans = 0.05
    ytrans = 0.05
    # A.Rotate(border_mode=cv2_BORDER_CONSTANT,value=(255,255,255),p=1),
    ssr = A.ShiftScaleRotate(shift_limit=xtrans, scale_limit=0, rotate_limit=rotate_deg, interpolation=1,
                             border_mode=cv2_BORDER_CONSTANT, value=(255, 255, 255), mask_value=(0, 0, 0),
                             shift_limit_x=None, shift_limit_y=None, always_apply=False, p=1)
    rbc = A.RandomBrightnessContrast(p=1)
    if keypoints is not None:
        keypoints = [keypoints]
        keypoint_params = A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True)
    else:
        keypoint_params = None

    transform = A.Compose([ssr, rbc], keypoint_params=keypoint_params)

    transformed = transform(image=img, mask=seg, keypoints=keypoints)
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"][0]
    transformed_seg = transformed["mask"]
    return transformed_image, transformed_seg, transformed_keypoints


def get_aug_data(IMAGE_SIZE):
    aug_data = SimpleNamespace()
    aug_data.color_add_range = int(0.2 * 255)
    aug_data.rotate_deg = 10
    aug_data.xtrans = 0.1 * IMAGE_SIZE[1]
    aug_data.ytrans = 0.05 * IMAGE_SIZE[0]
    aug_data.image_size = IMAGE_SIZE
    return aug_data