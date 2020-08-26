import torch
import torch.utils.data as data
import time
from tqdm import tqdm
from logging import getLogger

from bilstm_crf_ner_pytorch.dataloader import Dataset, collate_fn
from bilstm_crf_ner_pytorch.utils import get_entities

logger = getLogger(__name__)


class Trainer(object):
    
    def __init__(self, model, optimizer, preprocessor, use_char=True, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.preprocessor = preprocessor
        self.use_char = use_char
        self.device = device
        self.best_model = None

    def train(self, x_train, y_train, x_valid=None, y_valid=None, epochs=150, batch_size=32, shuffle=True, early_stop=0):

        train_data = Dataset(x_train, y_train, preprocessor=self.preprocessor.transform)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

        if x_valid and y_valid:
            valid_data = Dataset(x_valid, y_valid, preprocessor=self.preprocessor.transform)
            valid_loader = data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

        start_at = time.time()
        best_loss = 1e+12
        for epoch in range(epochs):

            Loss = []
            self.model.train()
            for batch in tqdm(train_loader):
                loss = self.model(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

                Loss.append(loss.item())
                torch.cuda.empty_cache()

            ave_loss = sum(Loss) / len(Loss)
            temp_time = time.time()
            print(f'(Train-data) Epoch: {epoch + 1} Time: {temp_time - start_at} loss: {ave_loss}')
            logger.info(f'(Train_data) Epoch: {epoch + 1} Time: {temp_time - start_at} loss: {ave_loss}')

            if valid_loader:
                Loss = []
                num_pred = 0
                num_gold = 0
                num_tp = 0
                self.model.eval()
                for batch in tqdm(valid_loader):
                    with torch.no_grad():
                        _, _, lengths, _, _, _ = batch
                        loss = self.model(batch)
                        pred_paths, gold_tags = self.model.decode_tags(batch)
                        pred_tags, pred_scores = zip(*pred_paths)
                        pred_chunks = set(get_entities(pred_tags, self.preprocessor, lengths))
                        gold_chunks = set(get_entities(gold_tags, self.preprocessor, lengths))

                        num_pred += len(pred_chunks)
                        num_gold += len(gold_chunks)
                        num_tp += len(pred_chunks & gold_chunks)

                    Loss.append(loss.item())

                precision = num_tp / num_pred if num_pred > 0 else 0
                recall = num_tp / num_gold if num_gold > 0 else 0
                f1score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                print(f'(Dev data) Loss: {sum(Loss)/len(Loss)} Precision: {precision} Recall: {recall} F1Score: {f1score}')
                logger.info(f'(Dev data) Loss: {sum(Loss)/len(Loss)} Precision: {precision} Recall: {recall} F1Score: {f1score}')
                if sum(Loss)/len(Loss) < best_loss:
                    stop_count = 0
                    print('best_model update')
                    logger.info(f'model update')
                    self.best_model = self.model.state_dict()
                else:
                    stop_count += 1
                    if early_stop > 0 and stop_count > early_stop:
                        print(f'Early stopping at epoch {epoch+1}')
                        logger.info(f'Early stoppping at epoch {epoch+1}')
