import torch
from torch.utils.data import Dataset
from collections import Counter
import nltk
import os
from PIL import Image


def tokenize_text(text, vocab, max_len=10):
    tokens = nltk.word_tokenize(text.lower())
    token_ids = [vocab.get(token, vocab['<pad>']) for token in tokens]

    if len(token_ids) < max_len:
        token_ids += [vocab['<pad>']] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]

    return token_ids


class BaseDataset(Dataset):
    def __init__(self, df, root_dir, image_transform, vocab, max_len):
        self.csv_data = df
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return self.csv_data.shape[0]

    def text_tokenization(self, sent):
        input_ids = tokenize_text(sent, self.vocab, self.max_len)
        input_ids = torch.tensor(input_ids)
        return input_ids

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be overridden by child class")


class WeiboDataset4Cnn(BaseDataset):
    def __init__(self, df, root_dir, image_transform, vocab, max_len, processor):
        super().__init__(df, root_dir, image_transform, vocab, max_len)
        self.processor = processor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.csv_data['label'][idx]
        label = int(label)
        if label == 1:  # fake
            image_folder = self.root_dir + "rumor_images/"
        else:
            image_folder = self.root_dir + "nonrumor_images/"
        label = torch.tensor(label)

        image_name = image_folder + self.csv_data['image_id'][idx]
        if not os.path.exists(image_name):
            print("ERROR: image_name not exists")
        image = Image.open(image_name).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        image_token = inputs["pixel_values"].squeeze()

        text = self.csv_data['content'][idx]
        input_ids = self.text_tokenization(text)

        sample = {
            'label': label,
            'image_token': image_token,
            'text_token': input_ids,
        }

        return sample

class PhemeDataset4Cnn(BaseDataset):
    def __init__(self, df, root_dir, image_transform, vocab, max_len, processor):
        super().__init__(df, root_dir, image_transform, vocab, max_len)
        self.processor = processor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.root_dir + str(self.csv_data['image_id'][idx])
        if not os.path.exists(image_name):
            print("ERROR: image_name not exists")
        image = Image.open(image_name).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        image_token = inputs["pixel_values"].squeeze()

        text = self.csv_data['content'][idx]
        input_ids = self.text_tokenization(text)

        label = self.csv_data['label'][idx]
        label = int(label)
        label = torch.tensor(label)

        sample = {
            'image_token': image_token,
            'text_token': input_ids,
            'label': label
        }

        return sample

