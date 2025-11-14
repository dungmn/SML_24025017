# save as reproduce_fig2.py
import os
import random
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import (
    vgg16, resnet50, inception_resnet_v2
)
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# --------------------------
# User settings
# --------------------------
IMAGENET_VAL_DIR = "./imagenet_val"  # directory with validation images
N_IMAGES = 1000                       # up to this many images (default paper used 1000)
RANDOM_SEED = 1234
OUTPUT_CSV = "fig2_results.csv"
# --------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --------------------------
# Utility helpers
# --------------------------
def pil_to_np(img_pil):
    return np.array(img_pil)

def center_crop_or_pad_to(img, target_w, target_h):
    # Resize image preserving aspect so min dimension >= requested, then center crop/pad
    iw, ih = img.size
    scale = max(target_w / iw, target_h / ih)
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img_resized.crop((left, top, left + target_w, top + target_h))

# Szegedy-like random crop (approx)
def random_szegedy_crop(img, out_size):
    # parameters similar to Inception training: area range [0.08,1.0], aspect [3/4,4/3]
    iw, ih = img.size
    for _ in range(50):
        area = iw * ih
        target_area = random.uniform(0.08, 1.0) * area
        aspect = random.uniform(3.0/4.0, 4.0/3.0)
        w = int(round((target_area * aspect) ** 0.5))
        h = int(round((target_area / aspect) ** 0.5))
        if w <= iw and h <= ih:
            x = random.randint(0, iw - w)
            y = random.randint(0, ih - h)
            crop = img.crop((x, y, x + w, y + h))
            return crop.resize((out_size, out_size), Image.BILINEAR), (x, y, w, h)
    # fallback to central crop
    return center_crop_or_pad_to(img, out_size, out_size), ( (iw - out_size)//2, (ih-out_size)//2, out_size, out_size )

# embedding: downscale so min dim = min_dim and embed into canvas (fill color)
def embed_scaled(img, canvas_size=224, min_dim=100, fill=(0,0,0), embed_x=None, embed_y=None):
    iw, ih = img.size
    scale = min_dim / min(iw, ih)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    small = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new('RGB', (canvas_size, canvas_size), fill)
    max_x = canvas_size - new_w
    max_y = canvas_size - new_h
    if embed_x is None:
        embed_x = random.randint(0, max_x if max_x > 0 else 0)
    if embed_y is None:
        embed_y = random.randint(0, max_y if max_y > 0 else 0)
    canvas.paste(small, (embed_x, embed_y))
    return canvas, (embed_x, embed_y, new_w, new_h)

# inpaint black pixels (simple using OpenCV)
def inpaint_black(pil_img):
    npim = np.array(pil_img)
    # mask: where pixel is exactly black
    mask = np.all(npim == [0,0,0], axis=2).astype('uint8') * 255
    try:
        inpainted = cv2.inpaint(npim[:,:,::-1], mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        inpainted_rgb = inpainted[:,:,::-1]
        return Image.fromarray(inpainted_rgb)
    except Exception as e:
        # fallback: simple Gaussian blur to fill
        blurred = cv2.GaussianBlur(npim, (7,7), 0)
        m = mask[:,:,None].astype(bool)
        npim[m] = blurred[m]
        return Image.fromarray(npim)

# protocol implementations
def protocol_crop_pair(img_pil, model_input_size):
    # sample random crop then shift the crop by one pixel diagonally in original image coords
    crop1, (x,y,w,h) = random_szegedy_crop(img_pil, out_size=model_input_size)
    # second crop: shift by +1,+1 inside original if possible, otherwise shift -1,-1
    iw, ih = img_pil.size
    dx = 1 if x + w + 1 <= iw else (-1 if x-1 >= 0 else 0)
    dy = 1 if y + h + 1 <= ih else (-1 if y-1 >= 0 else 0)
    x2 = x + dx
    y2 = y + dy
    crop2 = img_pil.crop((x2, y2, x2 + w, y2 + h)).resize((model_input_size, model_input_size), Image.BILINEAR)
    return crop1, crop2

def protocol_embedding_black_pair(img_pil, canvas_size=224, min_dim=100):
    # embed scaled image, then create second by shifting embedding location by 1 pixel
    canvas1, (ex, ey, nw, nh) = embed_scaled(img_pil, canvas_size=canvas_size, min_dim=min_dim)
    # shift embedding by 1 pixel (prefer +1)
    ex2 = ex + 1 if ex + 1 + nw <= canvas_size else (ex - 1 if ex - 1 >= 0 else ex)
    ey2 = ey + 1 if ey + 1 + nh <= canvas_size else (ey - 1 if ey - 1 >= 0 else ey)
    canvas2, _ = embed_scaled(img_pil, canvas_size=canvas_size, min_dim=min_dim, embed_x=ex2, embed_y=ey2)
    return canvas1, canvas2

def protocol_embedding_inpaint_pair(img_pil, canvas_size=224, min_dim=100):
    c1, c2 = protocol_embedding_black_pair(img_pil, canvas_size=canvas_size, min_dim=min_dim)
    # inpaint black regions
    ci1 = inpaint_black(c1)
    ci2 = inpaint_black(c2)
    return ci1, ci2

def protocol_scale_pair(img_pil, canvas_size=224, min_dim=100):
    # embed image with a size S, and make second with size S+1 (single-pixel change)
    iw, ih = img_pil.size
    # compute two scales such that min dim becomes min_dim and min_dim+1
    scale1 = min_dim / min(iw, ih)
    scale2 = (min_dim + 1) / min(iw, ih)
    w1, h1 = max(1, int(round(iw*scale1))), max(1, int(round(ih*scale1)))
    w2, h2 = max(1, int(round(iw*scale2))), max(1, int(round(ih*scale2)))
    small1 = img_pil.resize((w1, h1), Image.BILINEAR)
    small2 = img_pil.resize((w2, h2), Image.BILINEAR)
    # embed randomly
    max_x1 = canvas_size - w1
    max_y1 = canvas_size - h1
    ex1 = random.randint(0, max_x1 if max_x1 > 0 else 0)
    ey1 = random.randint(0, max_y1 if max_y1 > 0 else 0)
    # keep embedding location same for both (so only size diff)
    canvas1 = Image.new('RGB', (canvas_size, canvas_size), (0,0,0))
    canvas1.paste(small1, (ex1, ey1))
    # center second to same top-left if fits; otherwise adjust
    ex2 = min(ex1, canvas_size - w2) if canvas_size - w2 >= 0 else 0
    ey2 = min(ey1, canvas_size - h2) if canvas_size - h2 >= 0 else 0
    canvas2 = Image.new('RGB', (canvas_size, canvas_size), (0,0,0))
    canvas2.paste(small2, (ex2, ey2))
    return canvas1, canvas2

# --------------------------
# Model wrappers
# --------------------------
class ModelWrapper:
    def __init__(self, name):
        self.name = name
        if name == 'vgg16':
            self.model = vgg16.VGG16(weights='imagenet')
            self.input_size = 224
            self.preprocess = vgg16.preprocess_input
        elif name == 'resnet50':
            self.model = resnet50.ResNet50(weights='imagenet')
            self.input_size = 224
            self.preprocess = resnet50.preprocess_input
        elif name == 'inceptionresnetv2':
            self.model = inception_resnet_v2.InceptionResNetV2(weights='imagenet')
            self.input_size = 299
            self.preprocess = inception_resnet_v2.preprocess_input
        else:
            raise ValueError("unknown model")
    def predict_probs(self, pil_img):
        # pil_img should be RGB PIL at required input size
        img_arr = np.array(pil_img).astype('float32')
        # keras expects HxWxC
        x = np.expand_dims(img_arr, axis=0)
        x = self.preprocess(x)
        preds = self.model.predict(x, verbose=0)
        # ensure softmax (models already return logits->softmax in Keras applications)
        return preds[0]

# --------------------------
# Main experiment loop
# --------------------------
def run_experiment(models_to_run=['vgg16','resnet50','inceptionresnetv2']):
    # load image paths
    img_files = [os.path.join(IMAGENET_VAL_DIR, f) for f in os.listdir(IMAGENET_VAL_DIR)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
    img_files = sorted(img_files)[:N_IMAGES]
    print("Found %d images (using up to %d) in %s" % (len(img_files), N_IMAGES, IMAGENET_VAL_DIR))
    models = {name: ModelWrapper(name) for name in models_to_run}
    protocols = {
        'crop': protocol_crop_pair,
        'embedding_black': protocol_embedding_black_pair,
        'embedding_inpaint': protocol_embedding_inpaint_pair,
        'scale': protocol_scale_pair
    }

    # results store
    records = []
    for model_name, mw in models.items():
        for proto_name, proto_fn in protocols.items():
            top1_changes = []
            macs = []
            print(f"\nRunning model {model_name} protocol {proto_name} on {len(img_files)} images")
            for img_path in tqdm(img_files):
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print("skip", img_path, e)
                    continue

                # produce pair at the required pre-inference size for this model
                if proto_name == 'crop':
                    a, b = proto_fn(img_pil, mw.input_size)
                else:
                    # for embedding/scale we create 224x224 canvas then resize to model input
                    # so produce 224 and then resize if model expects different size
                    p1, p2 = proto_fn(img_pil, canvas_size=224, min_dim=100)
                    if mw.input_size != 224:
                        a = p1.resize((mw.input_size, mw.input_size), Image.BILINEAR)
                        b = p2.resize((mw.input_size, mw.input_size), Image.BILINEAR)
                    else:
                        a, b = p1, p2

                # get predictions
                probs_a = mw.predict_probs(a)
                probs_b = mw.predict_probs(b)
                top1_a = np.argmax(probs_a)
                top1_b = np.argmax(probs_b)
                p_change = int(top1_a != top1_b)
                # mean abs change for the top class (the class that had highest prob in frame A)
                mac = float(abs(float(probs_a[top1_a]) - float(probs_b[top1_a])))
                top1_changes.append(p_change)
                macs.append(mac)
                records.append({
                    'model': model_name,
                    'protocol': proto_name,
                    'image': os.path.basename(img_path),
                    'top1_a': int(top1_a),
                    'top1_b': int(top1_b),
                    'top1_change': p_change,
                    'mac': mac,
                })
            # aggregate
            p_top1_change = float(np.mean(top1_changes)) if len(top1_changes)>0 else 0.0
            mean_abs_change = float(np.mean(macs)) if len(macs)>0 else 0.0
            print(f"Model {model_name} Protocol {proto_name} -> P(top1 change) = {p_top1_change:.4f}, MAC = {mean_abs_change:.4f}")
    # save csv
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved detailed results to", OUTPUT_CSV)

if __name__ == "__main__":
    run_experiment()
