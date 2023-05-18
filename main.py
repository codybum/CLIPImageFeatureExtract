import argparse
import json
import sys, traceback
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, CLIPModel

def get_model_processor(args):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(args.model_path).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    return model, processor

def get_feature(model, processor, file):
    feature_array = None
    try:
        image = Image.open(file)
        inputs = processor(images=image, return_tensors="pt")

        image_features = model.get_image_features(**inputs)
        feature_array = image_features.cpu().detach().numpy()[0].tolist()

    except Exception:
        print('Unable to process:', file)
        traceback.print_exc(file=sys.stdout)

        
    return feature_array

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLIP Image Feature Extractor')
    parser.add_argument('--model_path', type=str, default='openai/clip-vit-base-patch32', help='name of project')
    parser.add_argument('--input_path', type=str, default='images', help='name of project')
    parser.add_argument('--output_path', type=str, default='extracts.json', help='name of project')

    # get args
    args = parser.parse_args()

    feature_extract_results = []

    # get extract list
    canidate_files = list(Path(args.input_path).rglob("*.*"))

    # if there are files in the list process them
    if len(canidate_files) > 0:

        #get models
        model,processor = get_model_processor(args)

        for file_path in canidate_files:
            file_path = str(file_path)
            #get features
            feature_array = get_feature(model, processor, file_path)
            if feature_array is not None:
                print('extracted features:', file_path)
                feature_data = dict()
                feature_data['file_path'] = file_path
                feature_data['features'] = feature_array
                feature_extract_results.append(feature_data)

    if len(feature_extract_results) > 0:

        # Serializing json
        json_object = json.dumps(feature_extract_results, indent=4)

        # Writing to sample.json
        with open(args.output_path, "w") as outfile:
            outfile.write(json_object)