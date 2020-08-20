import torch
import time
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score

from biltsm_crf_ner_pytorch.dataloader import NERSequence


class Trainer(object):
    
    def __init__(self, model, optimizer, preprocessor, use_char=True, gpu=False):
        self.model = model
        self.optimizer = optimizer
        self.preprocessor = preprocessor
        self.use_char = use_char
        self._gpu = gpu
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None,
             epochs=150, batch_size=32, shuffle=True):
        
        
        batch_train = NERSequence(x_train, y_train, batch_size=batch_size,
                                  preprocess=self.preprocessor.transform, shuffle=shuffle)

        if x_valid and y_valid:
            batch_valid = NERSequence(x_valid, y_valid, batch_size=batch_size, 
                                      preprocess=self.preprocessor.transform, shuffle=shuffle)

            
        #Check prediction before training
        with torch.no_grad():
            if self.use_char:
                precheck_x, precheck_x_char, precheck_y = batch_train[0][0][0], batch_train[0][0][1], batch_train[0][1]
                if self._gpu:
                    precheck_x = precheck_x.cuda()
                    precheck_x_char = precheck_x_char.cuda()
                    precheck_y = precheck_y.cuda()
            else:
                precheck_x, precheck_x_char, precheck_y = batch_train[0][0], None, batch_train[0][1]
                if self._gpu:
                    precheck_x = precheck_x.cuda()
                    precheck_y = precheck_y.cuda()
                
            print(self.model.neg_log_likelihood(precheck_x, precheck_y, char_sequence=precheck_x_char))
        

        start_at = time.time()
        best_loss = 100000
        
        if self.use_char:
            for epoch in range(epochs):
                loss_epoch = 0
                
                assert len(batch_train[0][0][0]) == len(batch_train[0][0][1]) and len(batch_train[0][0][0]) == len(batch_train[0][1])
                for i in tqdm(range(len(batch_train))):
                    sentence = batch_train[i][0][0]
                    char_sequence = batch_train[i][0][1]
                    tags = batch_train[i][1]
                    
                    if self._gpu:
                        sentence = sentence.cuda()
                        char_sequence = char_sequence.cuda()
                        tags = tags.cuda()
                    # Step 1. Remember that Pytorch accumulates gradients
                    self.model.zero_grad()

                    # Step 2. Run our forward pass.
                    loss = self.model.neg_log_likelihood(sentence, tags, char_sequence)

                    # Step 3. Compute the loss, gradients, and update the parameters
                    loss.backward()
                    self.optimizer.step()
                    
                    loss_epoch += loss
                loss_epoch /= len(batch_train) 

                if x_valid and y_valid:    
                    for i in tqdm(range(len(batch_valid))):
                        sentence = batch_valid[i][0][0]
                        char_sequence = batch_valid[i][0][1]
                        tags = batch_valid[i][1]

                    #f1 = F1score(valid_seq, preprocessor=self._preprocessor)
                    #callbacks = [f1] + callbacks if callbacks else [f1]
        else:
            for epoch in range(epochs):
                
                assert len(batch_train[0][0]) == len(batch_train[0][1])
                loss_epoch = 0
                for i in tqdm(range(len(batch_train))):
                    sentence = batch_train[i][0]
                    tags = batch_train[i][1]
                    
                    if self._gpu:
                        sentence = sentence.cuda()
                        tags = tags.cuda()

                    # Step 1. Remember that Pytorch accumulates gradients
                    self.model.zero_grad()

                    # Step 2. Run our forward pass.
                    loss = self.model.neg_log_likelihood(sentence, tags)
                    
                    # Step 3. Compute the loss, gradients, and update the parameters
                    loss.backward()
                    self.optimizer.step()
                    

                    loss_epoch += loss
                loss_epoch /= len(batch_train)
                
                if x_valid and y_valid:    
                    for i in tqdm(range(len(batch_valid))):
                        sentence = batch_valid[i][0]
                        tags = batch_valid[i][1]
                
                elapsed = time.time() - start_at
                print('-'*50)
                print(f'epoch:{epoch+1} loss: {loss_epoch[0]:.2f} elasped: {elapsed:.2f}')
                print('-'*50)
                #score, pred_tag_seq = self.model.forward(sentence)
                
                #if x_valid and y_valid:
                