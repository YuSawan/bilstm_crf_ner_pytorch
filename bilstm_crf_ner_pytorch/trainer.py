import torch
import torch.utils.data as data
import time
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score
from logging import getLogger

from bilstm_crf_ner_pytorch.dataloader import NERSequence, Dataset, collate_fn

logger = getLogger(__name__)

class Trainer(object):
    
    def __init__(self, model, optimizer, preprocessor, use_char=True, gpu=False):
        self.model = model
        self.optimizer = optimizer
        self.preprocessor = preprocessor
        self.use_char = use_char
        self._gpu = gpu

    def train(self, x_train, y_train, x_valid=None, y_valid=None,
             epochs=150, batch_size=32, shuffle=True):

        x_train, y_train = self.preprocessor.transform(x_train, y_train)
        train_data = Dataset(x_train, y_train)
        train_loader = data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

        if x_valid and y_valid:
            x_valid, y_valid = self.preprocessor.transform(x_valid, y_valid)
            valid_data = Dataset(x_valid, y_valid)
            valid_loader = data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)

        Loss = []
        for batch in valid_loader:
            with torch.no_grad():
                loss = self.model(batch)
                pred_tags, gold_tags = self.model.decode_tags(batch)
                for (pred_tag, pred_score), gold_tag in zip(pred_tags, gold_tags):
                    print(pred_tag)
                    print(gold_tag.tolist())
                    break
            Loss.append(loss.item())
            break

        '''
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
            logger.info(f'Epoch: {epoch + 1} Time: {temp_time - start_at} loss: {ave_loss}')

            if valid_loader:
                Loss = []
                self.model.eval()
                for batch in tqdm(valid_loader):
                    with torch.no_grad():
                        loss = self.model(batch)
                        pred_tags, gold_tags = self.model.decode_tags(batch)

                    Loss.append(loss.item())
                    torch.cuda.empty_cache()
                Loss = sum(Loss) / len(Loss)
        '''

