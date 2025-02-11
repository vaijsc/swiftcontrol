import os.path as osp
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

metadata = "../data/eval/MSCOCO/captions_val2014.json"
conditioning_path = "../data/eval/MSCOCO/conditioning"
output_file = "../data/eval/MSCOCO/processed_coco2.json"

# Load the annotations
with open(metadata, 'r') as f:
    anno = json.load(f)['annotations']
existed_img = []
# Function to process each image
def process_image(image):
    image_name = "COCO_val2014_000000" + str(image['image_id']) + '.jpg'
    if osp.exists(osp.join(conditioning_path, image_name)) and image_name not in existed_img:
        existed_img.append(image_name)
        return {"image_name": image_name, "text": image['caption']}
    return None

# Parallel processing
res = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_image, image) for image in anno]
    
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            res.append(result)

# Save the results into a JSON file
with open(output_file, 'w') as outfile:
    json.dump(res, outfile, indent=4)
print(f"Filtered annotations saved to {output_file}")
