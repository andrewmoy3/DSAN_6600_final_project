import os 
from PIL import Image
import tqdm

def resize_images_to_rgb():
    for i in range(2,13):
        src = f"data/images/images_{i:03d}/images"
        dst = f"data/images/images_{i:03d}/images"
        os.makedirs(dst, exist_ok=True)

        print(src, "->", dst)

        for root, _, files in os.walk(src):
            out = os.path.join(dst, os.path.relpath(root, src))
            os.makedirs(out, exist_ok=True)

            for f in tqdm.tqdm(files):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    img = Image.open(os.path.join(root, f)).convert("RGB")
                    img.resize((224, 224), Image.BILINEAR).save(os.path.join(out, f))
                
def augment_data():
        for i in range(1,13):
            src = f"data/images/images_{i:03d}/images"
            dst = f"data/images/images_{i+12:03d}/images"
            os.makedirs(dst, exist_ok=True)

            print(src, "->", dst)

            for root, _, files in os.walk(src):
                out = os.path.join(dst, os.path.relpath(root, src))
                os.makedirs(out, exist_ok=True)

                for f in tqdm.tqdm(files):
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        img = Image.open(os.path.join(root, f)).convert("RGB")
                        # flip img along y axis
                        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
                        img.save(os.path.join(out, f))
        
# augment_data()