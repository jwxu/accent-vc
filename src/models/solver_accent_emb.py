import os
import torch
import numpy as np
import time
import datetime
import wandb
import json
from collections import defaultdict, OrderedDict

from src.models.model_bl import D_ACCENT_VECTOR
from src.utils.eval import get_accuracy


class AccentEmbSolver(object):

    def __init__(self, train_data, config, val_data=None):
        """Initialize configurations."""

        self.config = config

        # Data loader.
        self.train_data = train_data
        self.val_data = val_data

        # Training configurations.
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.validation_freq = config.validation_freq
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Use pretrained checkpoint
        self.checkpoint = config.load_ckpt

        # Build the model and tensorboard.
        self.build_model()

        self.setup_logging()
        
            
    def build_model(self):
        """
        Build accent embedding model
        """
        self.model = D_ACCENT_VECTOR(dim_input=80, dim_cell=768, dim_emb=256, label_dim=6, classification=True)
        
        if self.checkpoint:
            print("Loading checkpoint...")
            checkpoint = torch.load(self.checkpoint, map_location=torch.device(self.device))
            
            if 'model_accent' in checkpoint.keys():
                self.model.load_state_dict(checkpoint['model_b'])
            else:
                new_state_dict = OrderedDict()
                for key, val in checkpoint['model_b'].items():
                    new_key = key[7:]
                    new_state_dict[new_key] = val
                self.model.load_state_dict(new_state_dict)
        
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 0.0001)
        

    def setup_logging(self):
        # Setup checkpoint directory
        checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists(self.config.checkpoints_dir):
            os.makedirs(self.config.checkpoints_dir)
        self.checkpoint_path = os.path.join(self.config.checkpoints_dir, checkpoint)
        os.makedirs(self.checkpoint_path)

        if self.config.wandb:
            # Set up wandb
            with open(self.config.wandb_json, 'r') as f:
                json_file = json.load(f)
                login_key = json_file['key']
                entity = json_file['entity']
            wandb.login(key=login_key)
            wandb.init(project=self.config.wandb, entity=entity)
            wandb.config.update(self.config)
            wandb.watch(self.model)
    

    def print_and_log(self, loss, epoch=None, total_steps=None, start_time=None):
        log = ''
        if start_time is not None:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            if total_steps is not None:
                log = "Elapsed [{}], Epoch [{}/{}][{}]".format(et, epoch, self.epochs, total_steps)
            else:
                log = "Elapsed [{}], Epoch [{}/{}]".format(et, epoch, self.epochs)

        stage = "Train" if epoch is not None else "Validation"
        label = "Step" if total_steps is not None else "Epoch"
    
        log_dict = {}
        for tag in loss.keys():
            avg_loss = sum(loss[tag]) / len(loss[tag])
            log += ", {}: {:.4f}".format(tag, avg_loss)
            log_dict[stage + "/" + tag + "/" + label] = avg_loss
        print(log)

        if self.config.wandb:
            i = total_steps
            if total_steps is None:
                i = epoch
            wandb.log(log_dict, i)
      
    
    #=====================================================================================================================================#
    
            
    def train(self):
        self.model.train()

        # Start training.
        print('Start training...')
        self.best_err = float("inf")
        total_steps = 0
        batch_loss = defaultdict(list)
        start_time = time.time()
        for epoch in range(self.epochs + 1):
            epoch_loss = defaultdict(list)
            for i, (data, labels) in enumerate(self.train_data):
                data = data.to(self.device)
                labels = labels.to(self.device)  
                                                        
                output = self.model(data)

                # Loss
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging.
                batch_loss['CE_Loss'].append(loss.item())
                epoch_loss['CE_Loss'].extend(batch_loss['CE_Loss'])

                total_steps += 1

                # Print out training information.
                if total_steps % self.log_step == 0:
                    self.print_and_log(batch_loss, epoch=epoch, total_steps=total_steps, start_time=start_time)
                    batch_loss = defaultdict(list)

            # Print out training information.
            self.print_and_log(epoch_loss, epoch=epoch, total_steps=None, start_time=start_time)

            # Validate
            if self.validation_freq > 0 and (epoch % self.validation_freq) == 0:
                self.validate(total_steps)
        
        checkpoint_path = os.path.join(self.checkpoint_path, 'accentemb.ckpt')
        print("Saving checkpoint to ", checkpoint_path)
        torch.save({'model_accent': self.model.state_dict()}, checkpoint_path)
    

    def validate(self, total_steps):
        """
        Validate model
        """
        self.model.eval()

        eval_metrics = defaultdict(list)
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.val_data):
                data = data.to(self.device)
                labels = labels.to(self.device)  
                                                        
                output = self.model(data)

                # Loss
                criterion = torch.nn.BCEWithLogitsLoss()
                loss = criterion(output, labels)
                acc = get_accuracy(output, labels)

                # Logging
                eval_metrics['CE_Loss'].append(loss.item())
                eval_metrics['Acc'].append(acc)
        
        self.print_and_log(eval_metrics, total_steps=total_steps)

        # Save the model that achieves the smallest validation error
        avg_loss = np.mean(eval_metrics['CE_Loss'])
        if avg_loss < self.best_err:
            self.best_err = avg_loss
            print(f'Saving best validation model with validation error {avg_loss:.7f}')

            checkpoint_path = os.path.join(self.checkpoint_path, 'accentemb_best.ckpt')
            print("Saving checkpoint to ", checkpoint_path)
            torch.save({'model_accent': self.model.state_dict()}, checkpoint_path)
                

