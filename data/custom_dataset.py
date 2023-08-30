import torch
import os
import numpy as np
from torchvision import transforms
import cv2
import PIL
from PIL import Image
import copy
from torchvideotransforms import video_transforms, volume_transforms, functional
import random

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, stage):
        super(VideoDataset, self).__init__()
        self.data = data
        self.args = args
        self.stage = stage
        self.set_transform()
        self.images_path = os.path.join(args.dataset_root_path, 'frames')
        self.flow_path = os.path.join(self.args.dataset_root_path, 'flow')
        self.save_index = 0
    def __getitem__(self, index):
        video_ids = self.data['video_id'][index]
        ped_ids = self.data['ped_id'][index]
        frame_list = self.data['frame'][index][:self.args.observe_length] # return first 15 frames as observed
        bboxes = self.data['bbox'][index] # return all 60 frames #[:-1] # take first 15 frames as input
        intention_binary = self.data['intention_binary'][index] # all frames intentions are returned
        intention_prob = self.data['intention_prob'][index] # all frames, 3-dimension votes probability
        disagree_score = self.data['disagree_score'][index] # all scores for all frames are returned
        skeleton = self.data['skeleton'][index] 
        images, cropped_images, cropped_flows, bboxes_aug = self.load_images(video_ids, frame_list, bboxes)
        data = {
            'cropped_images': cropped_images,
            'images': images,
            'bboxes': bboxes,
            'intention_binary': intention_binary,
            'intention_prob': intention_prob,
            'frames': np.array([int(f) for f in frame_list]),
            'video_id': video_ids[0], 
            'ped_id': ped_ids[0],
            'disagree_score': disagree_score,
            'skeleton': skeleton,
            'cropped_flows': cropped_flows,
            'bboxes_aug': bboxes_aug
        }

        return data

    def __len__(self):
        return len(self.data['frame'])

    def load_images(self, video_ids, frame_list, bboxes):
        images = []
        cropped_images = []
        video_name = video_ids[0]
        cropped_flows = []
        bboxes_output = bboxes
        for i in range(len(frame_list)):
            frame_id = frame_list[i]
            bbox = bboxes[i]
            img_path = os.path.join(self.images_path, video_name, str(frame_id).zfill(3)+'.jpg')
            img = self.rgb_loader(img_path)

            bbox = self.jitter_bbox(img, [bbox], self.args.crop_mode, 2.0)[0]
            bbox = self.squarify(bbox, 1, img.shape[1])
            bbox = list(map(int, bbox[0:4]))

            cropped_img = Image.fromarray(img).crop(bbox)
            cropped_img = np.array(cropped_img)
            cropped_img = self.img_pad(cropped_img, mode='pad_resize', size=224) # return PIL.image type

            flow_data_name = f"{str(frame_id).zfill(3)}.npy"
            flow_data_path = os.path.join(self.flow_path, video_name, flow_data_name)
            flowmap = np.load(flow_data_path)  # Load the flowmap

            cropped_flow = Image.fromarray(flowmap).crop(bbox)
            cropped_flow = np.array(cropped_flow)
            cropped_flow = self.img_pad(cropped_flow, mode='pad_resize', size=224, type='flow') # return PIL.image type

            images.append(Image.fromarray(img))
            cropped_images.append(cropped_img)
            cropped_flows.append(cropped_flow) 
        images, bboxes_output = apply_resize(images, bboxes_output, (224,398))


        if self.stage == 'train':
            images, bboxes_output = apply_random_resize(images, bboxes_output, (1.2,1))
            images, bboxes_output = apply_random_crop(images, bboxes_output, (224,398))

        augmented_video_frames = self.transform(images)
        augmented_cropped_video_frames = self.cropped_transform(cropped_images)
        cropped_flows = volume_transforms.ClipToTensor()(cropped_flows)
        cropped_flows = video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(cropped_flows)
        return augmented_video_frames , augmented_cropped_video_frames, cropped_flows, bboxes_output # Time x Channel x H x W


    def rgb_loader(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def set_transform(self):
        # You can add more video-specific transformations as needed
        if self.stage == 'train':
            video_transform_list = [
                video_transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ]
            self.transform = video_transforms.Compose(video_transform_list)
            cropped_video_transform_list = [
                video_transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ]
            self.cropped_transform = video_transforms.Compose(cropped_video_transform_list)
        else:
            # For validation/testing, you might want to have deterministic transformations
            video_transform_list = [
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ]
            self.transform = video_transforms.Compose(video_transform_list)
            cropped_video_transform_list = [
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ]
            self.cropped_transform = video_transforms.Compose(cropped_video_transform_list)

    def squarify(self, bbox, squarify_ratio, img_width):
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * squarify_ratio - width
        # width_change = float(bbox[4])*self._squarify_ratio - float(bbox[3])
        bbox[0] = bbox[0] - width_change/2
        bbox[2] = bbox[2] + width_change/2
        if bbox[0] < 0:
            bbox[0] = 0

        if bbox[2] > img_width:
            # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
            bbox[0] = bbox[0]-bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    def jitter_bbox(self, img, bbox, mode, ratio):
        assert (mode in ['same', 'enlarge', 'move', 'random_enlarge', 'random_move']), \
            'mode %s is invalid.' % mode

        if mode == 'same':
            return bbox


        if mode in ['random_enlarge', 'enlarge']:
            jitter_ratio = abs(ratio)
        else:
            jitter_ratio = ratio

        if mode == 'random_enlarge':
            jitter_ratio = np.random.random_sample() * jitter_ratio
        elif mode == 'random_move':
            jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

        jit_boxes = []
        for b in bbox:
            bbox_width = b[2] - b[0]
            bbox_height = b[3] - b[1]

            width_change = bbox_width * jitter_ratio
            height_change = bbox_height * jitter_ratio

            if width_change < height_change:
                height_change = width_change
            else:
                width_change = height_change

            if mode in ['enlarge', 'random_enlarge']:
                b[0] = b[0] - width_change // 2
                b[1] = b[1] - height_change // 2
            else:
                b[0] = b[0] + width_change // 2
                b[1] = b[1] + height_change // 2

            b[2] = b[2] + width_change // 2
            b[3] = b[3] + height_change // 2

            # Checks to make sure the bbox is not exiting the image boundaries
            b = self.bbox_sanity_check(img, b)
            jit_boxes.append(b)
        # elif crop_opts['mode'] == 'border_only':
        return jit_boxes

    def bbox_sanity_check(self, img, bbox):
        img_heigth, img_width, channel = img.shape
        if bbox[0] < 0:
            bbox[0] = 0.0
        if bbox[1] < 0:
            bbox[1] = 0.0
        if bbox[2] >= img_width:
            bbox[2] = img_width - 1
        if bbox[3] >= img_heigth:
            bbox[3] = img_heigth - 1
        return bbox

    def img_pad(self, img, mode='warp', size=224, type='image'):
        image = img.copy()
        img_size = (image.shape[0], image.shape[1]) # size is in (width, height)
        ratio = float(size) / max(img_size)
        if mode == 'pad_resize' or \
                (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = (int(img_size[0] * ratio), int(img_size[1] * ratio))# tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
            image = Image.fromarray(image)
            image = image.resize(img_size, PIL.Image.NEAREST)
        if type == 'image':
            padded_image = PIL.Image.new("RGB", (size, size))
        else:
            padded_image = PIL.Image.new("RGB", (size, size), color=(255, 255, 255))
        padded_image.paste(image, ((size - img_size[0]) // 2,
                                    (size - img_size[1]) // 2))
        return padded_image

def apply_random_crop(image, bbox, crop_size):
    im_w, im_h = image[0].size
    h, w = crop_size
    # Random 
    x1 = random.randint(0, im_w - w)
    y1 = random.randint(0, im_h - h)
    cropped_image = functional.crop_clip(image, y1, x1, h, w)
    bbox = np.array(bbox)
    resized_bbox = np.zeros_like(bbox)
    resized_bbox[:, 0] = np.maximum(0, bbox[:, 0] - x1)
    resized_bbox[:, 1] = np.maximum(0, bbox[:, 1] - y1)
    resized_bbox[:, 2] = np.minimum(w, bbox[:, 2] - x1)
    resized_bbox[:, 3] = np.minimum(h, bbox[:, 3] - y1)
    return cropped_image, resized_bbox

def apply_random_resize(image, bbox, ratio=(3./4., 4./3.)):
    im_w, im_h = image[0].size

    scaling_factor = random.uniform(ratio[0], ratio[1])
    new_w = int(im_w * scaling_factor)
    new_h = int(im_h * scaling_factor)
    new_size = (new_h, new_w)
    resized_image = functional.resize_clip(
            image, new_size, interpolation='bilinear')
    x_scale = new_w / im_w
    y_scale = new_h / im_h
    bbox = np.array(bbox)
    resized_bbox = np.zeros_like(bbox)
    resized_bbox[:, 0] = bbox[:, 0] * x_scale
    resized_bbox[:, 1] = bbox[:, 1] * y_scale
    resized_bbox[:, 2] = bbox[:, 2] * x_scale
    resized_bbox[:, 3] = bbox[:, 3] * y_scale

    return resized_image, resized_bbox

def apply_resize(image, bbox, size=(224, 398)):
    im_w, im_h = image[0].size

    resized_image = functional.resize_clip(
            image, size, interpolation='bilinear')
    x_scale = size[1] / im_w
    y_scale = size[0] / im_h
    bbox = np.array(bbox)
    resized_bbox = np.zeros_like(bbox)
    resized_bbox[:, 0] = bbox[:, 0] * x_scale
    resized_bbox[:, 1] = bbox[:, 1] * y_scale
    resized_bbox[:, 2] = bbox[:, 2] * x_scale
    resized_bbox[:, 3] = bbox[:, 3] * y_scale

    return resized_image, resized_bbox