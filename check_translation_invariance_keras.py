#!/usr/bin/env python3
"""
check_translation_invariance_keras.py

Keras/TensorFlow equivalent of check_translation_invariance.py (PyTorch).

This script loads ImageNet-pretrained models from tf.keras.applications and
measures translation-induced top-1 changes (jaggedness), mean absolute change
in predicted top-1 probability, and accuracy on the validation set, using
the same two protocols as the original script:
 - cropping protocol: when the prepared image is larger than 224, take
   224x224 crops shifted by 1 pixel and compare predictions
 - black-background protocol: when the prepared image is smaller than 224,
   embed it in a 224x224 black canvas and shift by 1 pixel

The input transformation is a pragmatic approximation of the PyTorch
RandomResizedCrop behavior used in the original script.

Usage:
    python check_translation_invariance_keras.py /path/to/datasets --arch resnet50

Dependencies: tensorflow (>=2.0), pillow, numpy
"""

import argparse
import os
import math
import sys
import numpy as np
from PIL import Image
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Keras translation invariance check')
    parser.add_argument('data', help='path to dataset (expects train/ and val/ subfolders like ImageFolder)')
    parser.add_argument('-a', '--arch', default='resnet50', choices=['resnet50', 'vgg16', 'densenet201'],
                        help='model architecture')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use (optional)')
    parser.add_argument('--pretrained', action='store_true', default=True, help='use pretrained imagenet weights')
    parser.add_argument('--max-images', default=1000, type=int, help='max images per size to process')
    return parser.parse_args()


def set_gpu(gpu):
    if gpu is not None:
        # set visible device to selected GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def get_tf_and_keras():
    """Import TensorFlow and Keras after CUDA_VISIBLE_DEVICES is set."""
    import tensorflow as tf
    from tensorflow import keras
    return tf, keras


def get_model_and_preprocess(arch):
    # Import TF/Keras here so that GPU visibility can be configured first.
    tf, keras = get_tf_and_keras()
    if arch == 'resnet50':
        from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
        model = ResNet50(weights='imagenet')
    elif arch == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
        model = VGG16(weights='imagenet')
    elif arch == 'densenet201':
        from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input
        model = DenseNet201(weights='imagenet')
    else:
        raise ValueError('Unsupported arch: %s' % arch)
    return model, preprocess_input


def build_class_index(valdir):
    # mimic PyTorch ImageFolder: classes are sorted folder names
    classes = [d for d in os.listdir(valdir) if os.path.isdir(os.path.join(valdir, d))]
    classes.sort()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # collect (image_path, class_idx)
    items = []
    for c in classes:
        folder = os.path.join(valdir, c)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                items.append((os.path.join(folder, fname), class_to_idx[c]))
    return items


def load_image_as_rgb(path):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def prepare_image_for_size(img, s):
    """
    Prepare an image similar to the PyTorch transformations used in the
    original script. For s <= 1.1 we scale the image by sqrt(s) (approx area
    fraction) and then center-crop or pad to 234x234. For s > 1.1 we resize
    the image so that its max dimension becomes int(224/s) (which will be
    <224), matching the small-image (black background) protocol.
    """
    w, h = img.size
    if s <= 1.1:
        # scale by sqrt(area_fraction)
        scale = math.sqrt(float(s))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_s = img.resize((new_w, new_h), Image.BILINEAR)
        # center crop or pad to 234x234
        target = 234
        if new_w >= target and new_h >= target:
            left = (new_w - target) // 2
            top = (new_h - target) // 2
            img_c = img_s.crop((left, top, left + target, top + target))
        else:
            # pad to target
            canvas = Image.new('RGB', (target, target), (0, 0, 0))
            left = max(0, (target - new_w) // 2)
            top = max(0, (target - new_h) // 2)
            canvas.paste(img_s, (left, top))
            img_c = canvas
        return np.array(img_c)
    else:
        small = int(max(1, round(224.0 / float(s))))
        img_s = img.resize((small, small), Image.BILINEAR)
        return np.array(img_s)


def jaggedness(items, model, preprocess_input, size_s, max_images=1000):
    # items: list of (path, target_idx)
    Jagged = 0.0
    MAC = 0.0
    Counter = 0.0
    Accc = 0.0
    start = time.time()

    for i, (path, target) in enumerate(items):
        if i >= max_images:
            break
        img = load_image_as_rgb(path)
        prepared = prepare_image_for_size(img, size_s)
        # ensure HWC
        h, w = prepared.shape[0], prepared.shape[1]

        if (w - 224 - 1) // 2 > 0:
            # cropping protocol
            center = (w - 224 - 1) // 2
            for c in range(center - 4, center + 4):
                x1 = c
                x2 = c + 224
                y1 = c
                y2 = c + 224
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    # if crop goes out of bounds, skip
                    continue
                crop1 = prepared[y1:y2, x1:x2, :]
                crop2 = prepared[y1 + 1:y2 + 1, x1 + 1:x2 + 1, :]

                batch1 = np.expand_dims(crop1.astype('float32'), axis=0)
                batch2 = np.expand_dims(crop2.astype('float32'), axis=0)
                batch1 = preprocess_input(batch1)
                batch2 = preprocess_input(batch2)

                out1 = model.predict(batch1, verbose=0)
                out2 = model.predict(batch2, verbose=0)
                pred1 = np.argmax(out1, axis=1)[0]
                pred2 = np.argmax(out2, axis=1)[0]
                Accc += float(pred1 == target)
                prob1 = float(out1[0, pred1])
                prob2 = float(out2[0, pred1])
                MAC += abs(prob1 - prob2)
                Jagged += float(pred1 != pred2)
                Counter += 1.0
        else:
            # black background protocol
            small_h, small_w = h, w
            center = (224 - small_w - 1) // 2
            for c in range(center - 4, center + 4):
                x1 = c
                y1 = c
                canvas1 = np.zeros((224, 224, 3), dtype=np.uint8)
                canvas2 = np.zeros((224, 224, 3), dtype=np.uint8)
                if x1 < 0 or y1 < 0 or (x1 + small_w) > 224 or (y1 + small_h) > 224:
                    continue
                canvas1[y1:y1 + small_h, x1:x1 + small_w, :] = prepared
                canvas2[y1 + 1:y1 + 1 + small_h, x1 + 1:x1 + 1 + small_w, :] = prepared

                batch1 = np.expand_dims(canvas1.astype('float32'), axis=0)
                batch2 = np.expand_dims(canvas2.astype('float32'), axis=0)
                batch1 = preprocess_input(batch1)
                batch2 = preprocess_input(batch2)

                out1 = model.predict(batch1, verbose=0)
                out2 = model.predict(batch2, verbose=0)
                pred1 = np.argmax(out1, axis=1)[0]
                pred2 = np.argmax(out2, axis=1)[0]
                Accc += float(pred1 == target)
                prob1 = float(out1[0, pred1])
                prob2 = float(out2[0, pred1])
                MAC += abs(prob1 - prob2)
                Jagged += float(pred1 != pred2)
                Counter += 1.0

        if (i + 1) % 50 == 0:
            print('Processed', i + 1, 'images for size', size_s)

    if Counter == 0:
        return 0.0, 0.0, 0.0
    return Jagged / Counter, MAC / Counter, Accc / Counter


def main():
    args = parse_args()
    set_gpu(args.gpu)
    valdir = os.path.join(args.data, 'val')
    if not os.path.isdir(valdir):
        print('Val directory not found:', valdir)
        sys.exit(1)

    items = build_class_index(valdir)
    print('Found %d validation images' % len(items))

    model, preprocess_input = get_model_and_preprocess(args.arch)
    # warm-up (build model)
    _ = model.predict(np.zeros((1, 224, 224, 3), dtype='float32'))

    Sizes = np.arange(0.1, 2.1, 0.3)
    results = []
    for s in Sizes:
        print('Evaluating size s =', float(s))
        j, m, a = jaggedness(items, model, preprocess_input, float(s), max_images=args.max_images)
        print('s=%.3f jagged=%.4f mac=%.4f acc=%.4f' % (s, j, m, a))
        results.append((j, m, a))

    results = np.array(results)
    np.save('jaggArchs_keras_%s.npy' % args.arch, results)
    print('Saved results to jaggArchs_keras_%s.npy' % args.arch)


if __name__ == '__main__':
    main()
