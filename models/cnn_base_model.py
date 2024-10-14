import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from transformers import ViTModel

from utils import Averager
from .layers import *


class TextEncoder(nn.Module):
    def __init__(self, args, fine_tune_module=False):
        super(TextEncoder, self).__init__()
        self.args = args
        self.embedding_dim = 300
        self.kernel_sizes = [3, 4, 5]
        self.num_filters = 256
        self.embedding = nn.Embedding(self.args.vocab_size, self.embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, (k, self.embedding_dim)) for k in self.kernel_sizes
        ])
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.num_filters, 128)
        self.fc2 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # [batch_size, seq_length, embedding_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_length, embedding_dim]
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [batch_size, num_filters, seq_length]

        if attention_mask is not None:
            for i in range(len(conv_outs)):
                conv_outs[i] = conv_outs[i] * attention_mask.unsqueeze(1)

        pooled = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outs]  # [batch_size, num_filters]
        cat = torch.cat(pooled, 1)  # [batch_size, num_filters * len(kernel_sizes)]
        x = self.dropout(F.gelu(self.fc1(cat)))
        out = self.dropout(F.gelu(self.fc2(x)))
        return out  # [batch_size, 32]


class VisionEncoder(nn.Module):
    def __init__(self, args, fine_tune_module=False):
        super(VisionEncoder, self).__init__()
        self.args = args
        self.fine_tune_module = fine_tune_module

        # self.vis_encoder = ViTModel.from_pretrained('../pretrained_models/google_vit-base-patch16-224-in21k').requires_grad_(False)
        self.vis_encoder = ViTModel.from_pretrained('../pretrained_models/google_vit-base-patch16-384').requires_grad_(
            False)

        self.vis_enc_fc1 = torch.nn.Linear(768, 2742)
        self.vis_enc_fc2 = torch.nn.Linear(2742, 32)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, images):
        vit_feature = self.vis_encoder(images).last_hidden_state[:, 0]
        x = self.dropout(F.gelu(self.vis_enc_fc1(vit_feature)))
        x = self.dropout(F.gelu(self.vis_enc_fc2(x)))
        return x


class CnnBaseModel(nn.Module):
    def __init__(self, args, feature_dim=64):
        super(CnnBaseModel, self).__init__()
        self.args = args
        self.text_encoder = TextEncoder(self.args)
        self.vision_encoder = VisionEncoder(self.args)

        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, input_ids, image):
        text_features = self.text_encoder(input_ids)
        image_features = self.vision_encoder(image)
        combined_features = torch.cat([text_features, image_features], dim=1)
        x = self.dropout(F.gelu(self.fc1(combined_features)))
        prediction = torch.sigmoid(self.fc2(x))
        prediction = prediction.squeeze(-1)
        prediction = prediction.float()
        return prediction


class Trainer():
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        self.model = CnnBaseModel(self.args)

        print(f"CUDA_VISIBLE_DEVICES:{self.args.gpu}")
        if torch.cuda.is_available():
            self.model.cuda()
            device = torch.device("cuda")
            print("CUDA will be used in train")
        else:
            device = 'cpu'
            print("CUDA will not be used in train")

        loss_fn = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.model.parameters())), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)

        print("-" * 20, "开始训练", "-" * 20)
        best_test_acc = 0.000
        patience = self.args.early_stop
        epochs_wo_improvement = 0
        train_loss_history = []
        train_acc_history = []
        test_loss_history = []
        test_acc_history = []
        # =======================================
        #               Epoch Loop
        # =======================================
        for epoch in range(self.args.epochs):
            epoch_time = time.time()
            epoch_average_loss = Averager()
            self.model.train()
            train_true = []
            train_pred = []
            # =======================================
            #           Train Batch Loop
            # =======================================
            for step, batch in enumerate(self.train_loader):
                # batch_average_loss = Averager()
                text_token = batch['text_token']
                image_token = batch['image_token']
                label = batch['label']
                b_input_ids = text_token.to(device)
                b_image = image_token.to(device)
                b_label = label.to(device)

                optimizer.zero_grad()
                logits = self.model(b_input_ids, b_image)

                b_label = b_label.to(torch.float32)
                loss = loss_fn(logits, b_label)
                epoch_average_loss.add(loss.item())

                loss.backward()
                optimizer.step()

                logits[logits < 0.5] = 0
                logits[logits >= 0.5] = 1
                train_true.append(label.detach().cpu().numpy())
                train_pred.append(logits.detach().cpu().numpy())

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_acc = metrics.accuracy_score(train_true, train_pred)
            train_acc_history.append(train_acc)
            epoch_time_elapsed = time.time() - epoch_time
            train_loss_history.append(epoch_average_loss.item())
            print(
                f"Epoch [{epoch + 1}/{self.args.epochs}], Loss: {epoch_average_loss.item():.4f}, Train_Acc: {train_acc:.4f}, Time_Elapsed: {epoch_time_elapsed:.2f}.")

            # =======================================
            #               Evaluation
            # =======================================
            self.model.eval()
            test_true = []
            test_pred = []
            test_average_loss = Averager()

            # =======================================
            #           Test Batch Loop
            # =======================================
            for step, batch in enumerate(self.test_loader):
                text_token = batch['text_token']
                image_token = batch['image_token']
                label = batch['label']
                b_input_ids = text_token.to(device)
                b_image = image_token.to(device)
                b_label = label.to(device)

                with torch.no_grad():
                    logits = self.model(b_input_ids, b_image)

                b_label = b_label.to(torch.float32)
                loss = loss_fn(logits, b_label)

                test_average_loss.add(loss.item())

                logits[logits < 0.5] = 0
                logits[logits >= 0.5] = 1
                test_true.append(label.detach().cpu().numpy())
                test_pred.append(logits.detach().cpu().numpy())

            test_loss_history.append(test_average_loss.item())
            print(f"Test average loss: {test_average_loss.item():.4f}")
            test_true = np.concatenate(test_true)  # test_true.shape: (1091,)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            test_acc_history.append(test_acc)
            print(f"Classification Acc: {test_acc:.4f}")
            print(
                f"Classification Report:\n"
                f"{metrics.classification_report(test_true, test_pred, digits=4, zero_division=1)}\n")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_wo_improvement = 0
            else:
                epochs_wo_improvement += 1

            if epochs_wo_improvement == patience:
                print('Early stopping at epoch {}...'.format(epoch + 1))
                break

        if self.args.save_history:
            dataset = self.args.dataset
            np.save('./history/cnn2/{}_train_loss_history.npy'.format(dataset),
                    np.array(train_loss_history))
            np.save('./history/cnn2/{}_train_acc_history.npy'.format(dataset), np.array(train_acc_history))
            np.save('./history/cnn2/{}_val_loss_history.npy'.format(dataset), np.array(test_loss_history))
            np.save('./history/cnn2/{}_val_acc_history.npy'.format(dataset), np.array(test_acc_history))
