import os
from src.models.model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import wandb
import json
from tqdm import tqdm


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        self.config = config

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Use pretrained checkpoint
        self.checkpoint = config.load_ckpt

        # Build the model and tensorboard.
        self.build_model()

        if config.wandb:
            self.setup_logging()
        
            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)      
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)
        
        if self.checkpoint:
            print("Loading checkpoint...")
            g_checkpoint = torch.load(self.checkpoint, map_location=torch.device(self.device))
            self.G.load_state_dict(g_checkpoint['model'])
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
    

    def setup_logging(self):
        # Set up wandb
        with open(self.config.wandb_json, 'r') as f:
            json_file = json.load(f)
            login_key = json_file['key']
            entity = json_file['entity']
        wandb.login(key=login_key)
        wandb.init(project=self.config.wandb, entity=entity)
        wandb.config.update(self.config)
        wandb.watch(self.G)
      
    
    #=====================================================================================================================================#
    
            
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real, emb_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, emb_org = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
            x_identic = x_identic.squeeze(1)
            x_identic_psnt = x_identic_psnt.squeeze(1)
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)

                log_dict = {}
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                    log_dict[tag] = loss[tag]
                print(log)

                if self.config.wandb:
                    wandb.log(log_dict, i)

        # Log final results
        et = time.time() - start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
        for tag in keys:
            log += ", {}: {:.4f}".format(tag, loss[tag])
        print(log)

        # Save model
        checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists(self.config.checkpoints_dir):
            os.makedirs(self.config.checkpoints_dir)
        checkpoint_path = os.path.join(self.config.checkpoints_dir, checkpoint)
        os.makedirs(checkpoint_path)
        
        checkpoint_path = os.path.join(checkpoint_path, 'accentvc.ckpt')
        print("Saving checkpoint to ", checkpoint_path)
        torch.save({'model': self.G.state_dict()}, checkpoint_path)
                

