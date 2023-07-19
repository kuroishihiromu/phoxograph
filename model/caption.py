import sys
import torch
import torchvision as tv
from transformers import BertTokenizer
from PIL import Image
import numpy as np
import subprocess

MAX_DIM = 299
max_position_embeddings=128
model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#キャプション生成の準備
def create_caption_and_mask():
    caption_template = torch.zeros((1,  max_position_embeddings), dtype=torch.long)
    mask_template = torch.ones((1,  max_position_embeddings), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask()


#画像入力
def input_image(path):
    image = Image.open(path)
    image = val_transform(image)
    image = image.unsqueeze(0)
    return image

@torch.no_grad()
def evaluate(image):
    model.eval()
    for i in range(max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption

#キャプション出力
def predict(image):
    output = evaluate(image)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    create_caption_and_mask()
    return result.capitalize()

if __name__ == "__main__":
    #path = input("パスを入力:")
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    image = input_image(path)
    result = predict(image)
    print(result)