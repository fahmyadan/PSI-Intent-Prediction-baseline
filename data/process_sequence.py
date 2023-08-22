import numpy as np
import json
import os
import pdb
import cv2
from data.custom_dataset import VideoDataset
from transformers import BertTokenizer, BertModel
import torch
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from models.pose_resnet import get_pose_net
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
POSE_RESNET_PATH = '/home/drisk/Downloads/pose_resnet_152_256x256.pth'
pose_resnet = get_pose_net(num_layers=152, init_weights=True, pretrained=POSE_RESNET_PATH)
pose_resnet = pose_resnet.to(device)
pose_resnet.eval()


def generate_data_sequence(set_name, database, database_driving, args):
    intention_prob = []
    intention_binary = []
    frame_seq = []
    pids_seq = []
    video_seq = []
    box_seq = []
    description_seq = []
    disagree_score_seq = []
    speed_seq = []
    skeleton_seq = []
    video_ids = sorted(database.keys())
    for video in sorted(video_ids): # video_name: e.g., 'video_0001'
        print(video)
        for ped in sorted(database[video].keys()): # ped_id: e.g., 'track_1'
            frame_seq.append(database[video][ped]['frames'])
            box_seq.append(database[video][ped]['cv_annotations']['bbox'])

            n = len(database[video][ped]['frames'])
            pids_seq.append([ped] * n)
            video_seq.append([video] * n)
            intents, probs, disgrs, descripts = get_intent(database, video, ped, args)
            intention_prob.append(probs)
            intention_binary.append(intents)
            disagree_score_seq.append(disgrs)
            description_seq.append(descripts)

            if video in database_driving:
                matching_indices = [i for i, frame in enumerate(database_driving[video]['frames']) if frame in database[video][ped]['frames']]
                matching_speeds = [database_driving[video]['speed'][i] for i in matching_indices]
            else:
                matching_speeds = [20.] * n
            speed_seq.append(matching_speeds)
            cropped_images_tensor = load_cropped_images(video, database[video][ped]['frames'], database[video][ped]['cv_annotations']['bbox'], args)
            # Split the tensor into batches of size 32
            batch_size = 32
            num_batches = len(cropped_images_tensor) // batch_size + (1 if len(cropped_images_tensor) % batch_size else 0)
            all_keypoints = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                cropped_images_batch = cropped_images_tensor[start_idx:end_idx].to(device)
                keypoints = pose_resnet(cropped_images_batch)
                keypoints_tensor = extract_keypoints_from_heatmaps(keypoints) * 4
                all_keypoints.append(keypoints_tensor.detach().cpu().numpy())
                
                # Optionally, you can visualize the results for the first image of each batch
                #plot_image_and_keypoints(cropped_images_batch[0], keypoints_tensor[0])
                torch.cuda.empty_cache()
            
            # Concatenate results from all batches
            all_keypoints = np.concatenate(all_keypoints, axis=0)
            skeleton_seq.append(all_keypoints)

    
    return {
        'frame': frame_seq,
        'bbox': box_seq,
        'intention_prob': intention_prob,
        'intention_binary': intention_binary,
        'ped_id': pids_seq,
        'video_id': video_seq,
        'disagree_score': disagree_score_seq,
        'description': description_seq,
        'speed' : speed_seq,
        'skeleton' : skeleton_seq
    }


def get_intent(database, video_name, ped_id, args):
    prob_seq = []
    intent_seq = []
    disagree_seq = []
    description_seq = []
    keyframe_seq = []
    n_frames = len(database[video_name][ped_id]['frames'])

    vid_uid_pairs = sorted((database[video_name][ped_id]['nlp_annotations'].keys()))
    n_users = len(vid_uid_pairs)
    for i in range(n_frames):
        labels = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['intent'][i] for vid_uid in vid_uid_pairs]
        descriptions = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['description'][i] for vid_uid in vid_uid_pairs]
        keyframe_seq = [database[video_name][ped_id]['nlp_annotations'][vid_uid]['key_frame'][i] for vid_uid in vid_uid_pairs]
        for j in range(len(labels)):
            if labels[j] == 'not_sure':
                labels[j] = 0.5
            elif labels[j] == 'not_cross':
                labels[j] = 0
            elif labels[j] == 'cross':
                labels[j] = 1
            else:
                raise Exception("Unknown intent label: ", labels[j])
            
        intent_prob = np.mean(labels)
        intent_binary = 0 if intent_prob < 0.5 else 1
        prob_seq.append(intent_prob)
        intent_seq.append(intent_binary)
        disagree_score = sum([1 if lbl != intent_binary else 0 for lbl in labels]) / n_users
        disagree_seq.append(disagree_score)

        descriptions = ''.join(descriptions)
        if descriptions.strip():  # Check if the description is not an empty string
            inputs = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad(): 
                outputs = model(**inputs)
            description_seq.append(np.array(outputs['pooler_output'][0].cpu()))
        else:
            description_seq.append(np.array(torch.zeros(model.config.hidden_size)))
    return intent_seq, prob_seq, disagree_seq, description_seq


def load_cropped_images(video_id, frame_list, bboxes, args):
    images_path = os.path.join(args.dataset_root_path, 'frames')
    cropped_images = []
    video_name = video_id

    for i in range(len(frame_list)):
        frame_id = frame_list[i]
        bbox = bboxes[i]
        img_path = os.path.join(images_path, video_name, str(frame_id).zfill(3)+'.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = jitter_bbox(img, [bbox], args.crop_mode, 2.0)[0]
        bbox = squarify(bbox, 1, img.shape[1])
        bbox = list(map(int, bbox[0:4]))

        cropped_img = Image.fromarray(img).crop(bbox)
        cropped_img = np.array(cropped_img)
        cropped_img = img_pad(cropped_img, mode='pad_resize')
        cropped_img = np.array(cropped_img)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        cropped_img = transform(cropped_img)
        cropped_images.append(cropped_img)

    return torch.stack(cropped_images)  # Time x Channel x H x W

def squarify(bbox, squarify_ratio, img_width):
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

def jitter_bbox(img, bbox, mode, ratio):

    if mode == 'same':
        return bbox

    # img = self.rgb_loader(img_path)
    # img_width, img_heigth = img.size

    if mode in ['random_enlarge', 'enlarge']:
        jitter_ratio = abs(ratio)
    else:
        jitter_ratio = ratio

    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample() * jitter_ratio
    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
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
        b = bbox_sanity_check(img, b)
        jit_boxes.append(b)
    # elif crop_opts['mode'] == 'border_only':
    return jit_boxes

def bbox_sanity_check(img, bbox):
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

def img_pad(img, mode='warp', target_width=256, target_height=256):

    image = img.copy()
    img_width = image.shape[1]
    img_height = image.shape[0]
    
    width_ratio = float(target_width) / img_width
    height_ratio = float(target_height) / img_height
    
    if mode == 'pad_resize' or \
            (mode == 'pad_fit' and (img_width > target_width or img_height > target_height)):
        new_width = int(img_width * min(width_ratio, height_ratio))
        new_height = int(img_height * min(width_ratio, height_ratio))
        image = Image.fromarray(image)
        image = image.resize((new_width, new_height), PIL.Image.NEAREST)

    padded_image = PIL.Image.new("RGB", (target_width, target_height))
    padded_image.paste(image, ((target_width - new_width) // 2,
                                (target_height - new_height) // 2))
    return padded_image


def extract_keypoints_from_heatmaps(heatmaps):
    # Get the x, y location of the max value for each heatmap
    max_val_coords = torch.argmax(heatmaps.view(heatmaps.shape[0], heatmaps.shape[1], -1), dim=2)
    
    # Convert the flattened index to 2D x, y coordinates
    keypoints_y, keypoints_x = max_val_coords // heatmaps.shape[3], max_val_coords % heatmaps.shape[3]
    
    # Stack the x, y coordinates
    keypoints = torch.stack((keypoints_x, keypoints_y), dim=2).float()
    
    return keypoints

def plot_image_and_keypoints(image_tensor, keypoints_tensor):
    # Convert image tensor to numpy array and de-normalize
    image_array = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_array = (image_array * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image_array = (image_array * 255).astype(np.uint8)

    # Extract the keypoints for the image
    keypoints_for_image = keypoints_tensor.cpu().numpy()

    # Plot the image and overlay the keypoints
    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.scatter(keypoints_for_image[:, 0], keypoints_for_image[:, 1], c='r', marker='o')
    plt.show()