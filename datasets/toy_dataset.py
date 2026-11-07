# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import io
import random
import argparse
try:
    import orjson as json
except ImportError:
    import json

from tqdm import tqdm
import pandas as pd 
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from torch.utils.data import ConcatDataset

from draw_marker import DRAW_FUNCTIONS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade

qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}

def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform

def collate_fn(batches, tokenizer):
    images = [_['images'] for _ in batches]
    questions = [_['question'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    indices = [_['index'] for _ in batches]
    return images, questions, answers, indices


class SPARDataset(torch.utils.data.Dataset):

    def __init__(self, meta, ds_name, repeat_time=1):
        with open(meta['annotation'], 'r') as f:
            self.data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.data = self.data[:int(len(self.data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.data = self.data * repeat_time

        self.root = meta['root']

    def __len__(self):
        return len(self.data)
    
    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path
    
    def load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def draw_image(self, image, data_item):
        draw_fn = DRAW_FUNCTIONS.get(data_item.get('type', None))
        if draw_fn is None:
            print(data_item)
            raise ValueError(f"Unsupported data type: {data_item.get('type', None)}")
        draw_fn(image, data_item)

    def multi_modal_get_item(self, data_item):

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        question = data_item['conversations'][0]['value']
        answer = data_item['conversations'][1]['value']
        # Merge the image path
        image_path = self.get_image_path(data_item['image'][0])

        image = self.load_image(image_path)
        self.draw_image(image, data_item)

        ret = dict(
            images=[image],
            question=question,
            answer=answer,
            index=data_item["id"]
        )
        return ret
    
    def multi_modal_multi_image_get_item(self, data_item):
        if '<image>' not in data_item['conversations'][0]['value']:
            num_image = len(data_item['image'])
            data_item['conversations'][0]['value'] = '<image>\n'*num_image + data_item['conversations'][0]['value']
        
        question = data_item['conversations'][0]['value']
        answer = data_item['conversations'][1]['value']

        images = []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            image_path = self.get_image_path(image_path)
            image = self.load_image(image_path)
            images.append(image)

        self.draw_image(images, data_item)
        
        ret = dict(
            images=images,
            question=question,
            answer=answer,
            index=data_item["id"]
        )
        return ret
    
    def fake_video_get_item(self, data_item):
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        images = []
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            images.append(image)
        self.draw_image(images, data_item)

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(images))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        question = data_item['conversations'][0]['value']
        answer = data_item['conversations'][1]['value']
        
        # Create the final return dictionary
        ret = dict(
            images=images,
            question=question,
            answer=answer,
            index=data_item["id"]
        )
        return ret

    def __getitem__(self, idx):
        data_item = json.loads(self.data[idx])
        if "image" in data_item and len(data_item["image"]) != 0:
            # for nav
            if data_item.get("type", None) == "nav":
                ret = self.fake_video_get_item(data_item)
            elif type(data_item["image"]) == list and len(data_item["image"]) > 1:
                # for multi image qa
                if len(data_item["image"]) <= 10:
                    ret = self.multi_modal_multi_image_get_item(data_item)
                # for our spar video qa
                else:
                    ret = self.fake_video_get_item(data_item)
            # for single image qa
            else:
                ret = self.multi_modal_get_item(data_item)
        elif "video" in data_item and len(data_item["video"]) != 0:
            ret = self.video_get_item(data_item)
        else:
            ret = self.pure_text_get_item(data_item)
        return ret

def build_datasets(
    json_dir,
):
    datasets = []
    lengths = []
    ds_collections = json.loads(open(json_dir).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        dataset = SPARDataset(
            ds_collections[ds_name],
            ds_name=ds_name,
            repeat_time=repeat_time,
        )
        datasets.append(dataset)
        lengths.append(len(dataset))

        print(f"Dataset {ds_name} has {len(dataset)} samples.")

    train_dataset = ConcatDataset(datasets)
    print("\n")
    print(f"Total number of samples: {sum(lengths)}")

    return train_dataset


def main(args: argparse.Namespace):
    random.seed(args.seed)
    
    dataset = build_datasets(
        json_dir=args.json_dir,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=None),
    )

    for _, (images, questions, answers, indices) in tqdm(enumerate(dataloader)):
        imgs = images[0]
        question = questions[0]
        answer = answers[0]
        index = indices[0]

        print(f"Sample #{index}")
        print("🟡 Question:\n", question)
        print("✅ Answer:\n", answer)

        num_views = len(imgs)
        fig, axs = plt.subplots(1, num_views, figsize=(5 * num_views, 5))

        if num_views == 1:
            axs = [axs]

        for i, img in enumerate(imgs):
            axs[i].imshow(img)
            axs[i].set_title(f"View {i}")
            axs[i].axis("off")

        plt.tight_layout()
        os.makedirs(args.image_save_dir, exist_ok=True)
        save_path = os.path.join(args.image_save_dir, f"sample_{index}.png")
        plt.savefig(save_path)
        plt.close(fig)

        print(f"🖼️ Saved visualization to {args.image_save_dir}")

        user_input = input("Press Enter to continue, or type 'q' to quit: ")
        if user_input.lower() == "q":
            break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=str, default='data_jsons/toy.json')
    parser.add_argument('--image-save-dir', type=str, default='debug_image')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
