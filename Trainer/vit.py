# Trainer for MaskGIT
import os
import random
import time
import math

import numpy as np
from tqdm import tqdm
from collections import deque
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from Trainer.trainer import Trainer
from Network.transformer import MaskTransformer

from Network.Taming.models.vqgan import VQModel


class MaskGIT(Trainer):

    def __init__(self, args):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__(args)
        self.args = args                                                        # Main argument see main.py
        self.scaler = torch.cuda.amp.GradScaler()                               # Init Scaler for multi GPUs
        # self.scaler = torch.amp.GradScaler('cuda')
        self.ae = self.get_network("autoencoder")
        self.codebook_size = self.ae.n_embed   
        print("Acquired codebook size:", self.codebook_size)   
        self.vit = self.get_network("vit")                                      # Load Masked Bidirectional Transformer   
        self.patch_size = self.args.img_size // 2**(self.ae.encoder.num_resolutions-1)     # Load VQGAN
        self.criterion = self.get_loss("cross_entropy", label_smoothing=0.1)    # Get cross entropy loss
        self.optim = self.get_optim(self.vit, self.args.lr, betas=(0.9, 0.96))  # Get Adam Optimizer with weight decay
        
        # Load data if aim to train or test the model
        if not self.args.debug:
            self.train_data, self.test_data = self.get_data()

        # Initialize evaluation object if testing
        if self.args.test_only:
            from Metrics.sample_and_eval import SampleAndEval
            self.sae = SampleAndEval(device=self.args.device, num_images=50_000)

    def get_network(self, archi):
        """ return the network, load checkpoint if self.args.resume == True
            :param
                archi -> str: vit|autoencoder, the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == "vit":
            model = MaskTransformer(
                img_size=self.args.img_size, hidden_dim=768, codebook_size=self.codebook_size, depth=24, heads=16, mlp_dim=3072, dropout=0.1     # Small
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=32, heads=16, mlp_dim=3072, dropout=0.1  # Big
                # img_size=self.args.img_size, hidden_dim=1024, codebook_size=1024, depth=48, heads=16, mlp_dim=3072, dropout=0.1  # Huge
            )

            if self.args.resume:
                ckpt = self.args.vit_folder
                ckpt += "current.pth" if os.path.isdir(self.args.vit_folder) else ""
                if self.args.is_master:
                    print("load ckpt from:", ckpt)
                # Read checkpoint file
                checkpoint = torch.load(ckpt, map_location='cpu')
                # checkpoint = torch.load(ckpt, map_location='cpu', weights_only=True)
                # Update the current epoch and iteration
                self.args.iter += checkpoint['iter']
                self.args.global_epoch += checkpoint['global_epoch']
                # Load network
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            model = model.to(self.args.device)
            if self.args.is_multi_gpus:  # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])

        elif archi == "autoencoder":
            # Load config
            # config = OmegaConf.load(self.args.vqgan_folder + "model.yaml")
            config = OmegaConf.load(self.args.vqgan_folder + "/model.yaml")
            model = VQModel(**config.model.params)
            checkpoint = torch.load(self.args.vqgan_folder + "/last.ckpt", map_location="cpu")["state_dict"]
            # checkpoint = torch.load(self.args.vqgan_folder + "/last.ckpt", map_location="cpu", weights_only=True)[
            #     "state_dict"]
            # Load network
            model.load_state_dict(checkpoint, strict=False)
            model = model.eval()
            model = model.to(self.args.device)
            

            if self.args.is_multi_gpus: # put model on multi GPUs if available
                model = DDP(model, device_ids=[self.args.device])
                model = model.module
        else:
            model = None

        if self.args.is_master:
            print(f"Size of model {archi}: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 10 ** 6:.3f}M")

        return model

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None, codebook_size=256):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        r = torch.rand(code.size(0))
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size)

        return mask_code, mask

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def train_one_epoch(self, log_iter=2500):
        """ Train the model for 1 epoch """
        self.vit.train()
        cum_loss = 0.
        window_loss = deque(maxlen=self.args.grad_cum)
        bar = tqdm(self.train_data, leave=False) if self.args.is_master else self.train_data
        n = len(self.train_data)
        # Start training for 1 epoch
        for x, y in bar:
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

            # Drop xx% of the condition for cfg
            drop_label = torch.empty(y.size()).uniform_(0, 1) < self.args.drop_label

            # VQGAN encoding to img tokens
            with torch.no_grad():
                emb, _, [_, _, code] = self.ae.encode(x)
                code = code.reshape(x.size(0), self.patch_size, self.patch_size)

            # Mask the encoded tokens
            masked_code, mask = self.get_mask_code(code, value=self.args.mask_value, codebook_size=self.codebook_size)

            # with torch.cuda.amp.autocast():                             # half precision
            with torch.amp.autocast('cuda'):  # half precision
                pred = self.vit(masked_code, y, drop_label=drop_label)  # The unmasked tokens prediction
                # Cross-entropy loss
                loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1)) / self.args.grad_cum

            # update weight if accumulation of gradient is done
            update_grad = self.args.iter % self.args.grad_cum == self.args.grad_cum - 1
            if update_grad:
                self.optim.zero_grad()

            self.scaler.scale(loss).backward()  # rescale to get more precise loss

            if update_grad:
                self.scaler.unscale_(self.optim)                      # rescale loss
                nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)  # Clip gradient
                self.scaler.step(self.optim)
                self.scaler.update()

            cum_loss += loss.cpu().item()
            window_loss.append(loss.data.cpu().numpy().mean())
            # logs
            if update_grad and self.args.is_master:
                self.log_add_scalar('Train/Loss', np.array(window_loss).sum(), self.args.iter)

            if self.args.iter % log_iter == 0 and self.args.is_master:
                # Generate sample for visualization
                gen_sample = self.sample(nb_sample=10)[0]
                gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Sampling", gen_sample, self.args.iter)
                # Show reconstruction
                unmasked_code = torch.softmax(pred, -1).max(-1)[1]
                reco_sample = self.reco(x=x[:10], code=code[:10], unmasked_code=unmasked_code[:10], mask=mask[:10])
                reco_sample = vutils.make_grid(reco_sample.data, nrow=10, padding=2, normalize=True)
                self.log_add_img("Images/Reconstruction", reco_sample, self.args.iter)

                # Save Network
                self.save_network(model=self.vit, path=self.args.vit_folder+"current.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            self.args.iter += 1

        return cum_loss / n

    def fit(self):
        """ Train the model """
        if self.args.is_master:
            print("Start training:")

        start = time.time()
        # Start training
        for e in range(self.args.global_epoch, self.args.epoch):
            # synch every GPUs
            if self.args.is_multi_gpus:
                self.train_data.sampler.set_epoch(e)

            # Train for one epoch
            train_loss = self.train_one_epoch()

            # Synch loss
            if self.args.is_multi_gpus:
                train_loss = self.all_gather(train_loss, torch.cuda.device_count())

            # Save model
            if e % 10 == 0 and self.args.is_master:
                self.save_network(model=self.vit, path=self.args.vit_folder + f"epoch_{self.args.global_epoch:03d}.pth",
                                  iter=self.args.iter, optimizer=self.optim, global_epoch=self.args.global_epoch)

            # Clock time
            clock_time = (time.time() - start)
            if self.args.is_master:
                self.log_add_scalar('Train/GlobalLoss', train_loss, self.args.global_epoch)
                print(f"\rEpoch {self.args.global_epoch},"
                      f" Iter {self.args.iter :},"
                      f" Loss {train_loss:.4f},"
                      f" Time: {clock_time // 3600:.0f}h {(clock_time % 3600) // 60:.0f}min {clock_time % 60:.2f}s")
            self.args.global_epoch += 1

    def eval(self):
        """ Evaluation of the model"""
        self.vit.eval()
        if self.args.is_master:
            print(f"Evaluation with hyper-parameter ->\n"
                  f"scheduler: {self.args.sched_mode}, number of step: {self.args.step}, "
                  f"softmax temperature: {self.args.sm_temp}, cfg weight: {self.args.cfg_w}, "
                  f"gumbel temperature: {self.args.r_temp}")
        # Evaluate the model
        m = self.sae.compute_and_log_metrics(self)
        self.vit.train()
        return m

    def reco(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """
        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                code = code.view(code.size(0), self.patch_size, self.patch_size)
                # Decoding reel code
                _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
                if mask is not None:
                    # Decoding reel code with mask to hide
                    mask = mask.view(code.size(0), 1, self.patch_size, self.patch_size).float()
                    __x2 = _x * (1 - F.interpolate(mask, (self.args.img_size, self.args.img_size)).to(self.args.device))
                    l_visual.append(__x2)
            if masked_code is not None:
                # Decoding masked code
                masked_code = masked_code.view(code.size(0), self.patch_size, self.patch_size)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0,  self.codebook_size-1))
                l_visual.append(__x)

            if unmasked_code is not None:
                # Decoding predicted code
                unmasked_code = unmasked_code.view(code.size(0), self.patch_size, self.patch_size)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
                l_visual.append(___x)

        return torch.cat(l_visual, dim=0)

    def sample(self, init_code=None, mask=None, orig_code = None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = mask.view(nb_sample, self.patch_size*self.patch_size)
                # mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                w = 0
                # with torch.cuda.amp.autocast():  # half precision
                with torch.amp.autocast('cuda'):  # half precision
                    if w != 0:
                        # Model Prediction
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0))
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / (len(scheduler)-1))
                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        # logit = self.vit(code.clone(), labels, drop_label=~drop)
                        # logit2 = self.vit(code.clone(), labels, drop_label=drop)
                        logit = self.vit(code.clone(), labels, drop_label=drop)

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                Ocode = orig_code.view(nb_sample, self.patch_size * self.patch_size)
                debug1 = pred_code[mask == 1]
                debug2 = Ocode[mask == 1]
                Corr_Rate = torch.sum(debug1-debug2 == 0)/debug1.size()[0]
                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))
                N1 = conf.squeeze(2)[mask == 1]
                N2 = debug1 - debug2
                result_matrix = torch.cat((N1.view(1, -1), N2.view(1, -1)), dim=0)
                N3 = result_matrix.cpu().numpy()
                # if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                #     ratio = (indice / (len(scheduler)-1))
                #     rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                #     conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                # elif randomize == "warm_up":  # chose random sample for the 2 first steps
                #     conf = torch.rand_like(conf) if indice < 2 else conf
                # elif randomize == "random":   # chose random prediction at each step
                #     conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                # tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            # decode the final prediction
            _code = torch.clamp(code, 0,  self.codebook_size-1)
            x = self.ae.decode_code(_code)

        self.vit.train()
        return x, l_codes, l_mask


    def GenDecodingNew(self, init_code=None, _mask=None, orig_code = None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        mask0 = _mask.clone()
        init_code0 = init_code.clone()
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = _mask.view(nb_sample, self.patch_size*self.patch_size)
                # mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)

            # 计算总token数量
            total_tokens = self.patch_size * self.patch_size
            
            # 修改：创建自定义scheduler，每次更新10%的tokens
            tokens_per_step = max(1, int(total_tokens * 0.1))  # 每步更新10%的token，至少1个
            max_steps = total_tokens // tokens_per_step + (1 if total_tokens % tokens_per_step > 0 else 0)
            # 如果用户指定的step小于我们计算的最大步数，则调整每步token数量
            if step < max_steps:
                tokens_per_step = max(1, total_tokens // step + (1 if total_tokens % step > 0 else 0))
                max_steps = step
                
            # 创建一个固定每步token数的简单scheduler
            custom_scheduler = [tokens_per_step] * max_steps
            # 最后一步可能不足tokens_per_step个
            remaining_tokens = total_tokens - tokens_per_step * (max_steps - 1)
            if remaining_tokens > 0 and remaining_tokens < tokens_per_step:
                custom_scheduler[-1] = remaining_tokens
                
            # 使用自定义scheduler代替原来的scheduler
            scheduler = custom_scheduler

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                print(f"Step {indice+1}/{len(scheduler)}, predicting {t} tokens")
                
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                # w = 0
                # with torch.cuda.amp.autocast():  # half precision
                with torch.amp.autocast('cuda'):  # half precision
                    if w == 9:
                        # Model Prediction
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0))
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / (len(scheduler)-1))
                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                    elif w == 0:  # no class / drop class
                        logit = self.vit(code.clone(), labels, drop_label=drop)
                    elif w == 1:  # with class / use class
                        logit = self.vit(code.clone(), labels, drop_label=~drop)

                    # if w != 0:
                    #     # Model Prediction
                    #     logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                    #                      torch.cat([labels, labels], dim=0),
                    #                      torch.cat([~drop, drop], dim=0))
                    #     logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    #     _w = w * (indice / (len(scheduler)-1))
                    #     # Classifier Free Guidance
                    #     logit = (1 + _w) * logit_c - _w * logit_u
                    # else:
                    #     # logit = self.vit(code.clone(), labels, drop_label=~drop)
                    #     # logit2 = self.vit(code.clone(), labels, drop_label=drop)
                    #     logit = self.vit(code.clone(), labels, drop_label=drop)

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                ############################################################################################################
                ### This sampling part can be replaced by argmax as follows
                # max_values, pred_code = torch.max(prob, dim=2)
                ############################################################################################################

                # Ocode = orig_code.view(nb_sample, self.patch_size * self.patch_size)
                # debug1 = pred_code[mask == 1]
                # debug2 = Ocode[mask == 1]
                # Corr_Rate = torch.sum(debug1-debug2 == 0)/debug1.size()[0]
                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))
                # conf = torch.log(conf.squeeze())

                # N1 = conf.squeeze(2)[mask == 1]
                # N2 = debug1 - debug2
                # result_matrix = torch.cat((N1.view(1, -1), N2.view(1, -1)), dim=0)
                # N3 = result_matrix.cpu().numpy()
                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                breakpoint()
                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())
                # print(t)


            # decode the final prediction
            _code = torch.clamp(code, 0,  self.codebook_size-1)
            x = self.ae.decode_code(_code)
            x_orig = self.ae.decode_code(orig_code)
            init_code0[mask0 == 1] = torch.randint(0, 1024, (torch.sum(mask0 == 1),), dtype=torch.int64).to(x_orig.device)
            x_random = self.ae.decode_code(init_code0)
            Ocode = orig_code.view(nb_sample, self.patch_size * self.patch_size)
            mask0 = mask0.view(nb_sample, self.patch_size*self.patch_size)
            debug2 = Ocode[mask0 == 1]
            debug1 = _code.view(nb_sample, self.patch_size*self.patch_size)[mask0 == 1]
            Correct_Rate = torch.sum(debug1-debug2 == 0)/debug1.size()[0]

            N_err = debug1.size()[0]
        # self.vit.train()
        return x, x_orig, x_random, l_codes, l_mask, Correct_Rate, N_err

    def GenDecoding(self, init_code=None, _mask=None, orig_code = None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        mask0 = _mask.clone()
        init_code0 = init_code.clone()
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = _mask.view(nb_sample, self.patch_size*self.patch_size)
                # mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                # print(t)
                # print(indice)
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                # w = 0
                # with torch.cuda.amp.autocast():  # half precision
                with torch.amp.autocast('cuda'):  # half precision
                    if w == 9:
                        # Model Prediction
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0))
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / (len(scheduler)-1))
                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                    elif w == 0:  # no class / drop class
                        logit = self.vit(code.clone(), labels, drop_label=drop)
                    elif w == 1:  # with class / use class
                        logit = self.vit(code.clone(), labels, drop_label=~drop)

                    # if w != 0:
                    #     # Model Prediction
                    #     logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                    #                      torch.cat([labels, labels], dim=0),
                    #                      torch.cat([~drop, drop], dim=0))
                    #     logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    #     _w = w * (indice / (len(scheduler)-1))
                    #     # Classifier Free Guidance
                    #     logit = (1 + _w) * logit_c - _w * logit_u
                    # else:
                    #     # logit = self.vit(code.clone(), labels, drop_label=~drop)
                    #     # logit2 = self.vit(code.clone(), labels, drop_label=drop)
                    #     logit = self.vit(code.clone(), labels, drop_label=drop)

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                ############################################################################################################
                ### This sampling part can be replaced by argmax as follows
                # max_values, pred_code = torch.max(prob, dim=2)
                ############################################################################################################

                # Ocode = orig_code.view(nb_sample, self.patch_size * self.patch_size)
                # debug1 = pred_code[mask == 1]
                # debug2 = Ocode[mask == 1]
                # Corr_Rate = torch.sum(debug1-debug2 == 0)/debug1.size()[0]
                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))
                # conf = torch.log(conf.squeeze())

                # N1 = conf.squeeze(2)[mask == 1]
                # N2 = debug1 - debug2
                # result_matrix = torch.cat((N1.view(1, -1), N2.view(1, -1)), dim=0)
                # N3 = result_matrix.cpu().numpy()
                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.args.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())
                # print(t)


            # decode the final prediction
            _code = torch.clamp(code, 0,  self.codebook_size-1)
            x = self.ae.decode_code(_code)
            x_orig = self.ae.decode_code(orig_code)
            init_code0[mask0 == 1] = torch.randint(0, 1024, (torch.sum(mask0 == 1),), dtype=torch.int64).to(x_orig.device)
            x_random = self.ae.decode_code(init_code0)
            Ocode = orig_code.view(nb_sample, self.patch_size * self.patch_size)
            mask0 = mask0.view(nb_sample, self.patch_size*self.patch_size)
            debug2 = Ocode[mask0 == 1]
            debug1 = _code.view(nb_sample, self.patch_size*self.patch_size)[mask0 == 1]
            Correct_Rate = torch.sum(debug1-debug2 == 0)/debug1.size()[0]
            # print(debug1.size())
            # breakpoint()
            TER_af = torch.sum(debug1-debug2 == 0)/nb_sample/self.patch_size/self.patch_size
            N_err = debug1.size()[0]
        # self.vit.train()
        return x, x_orig, x_random, l_codes, l_mask, TER_af, N_err
        
        # return None, None, None, l_codes, l_mask, Correct_Rate, N_err

############################ Here, I will change the code heavily, with options ################################
    def GenDecoding_options(self, init_code=None, options=None, _mask=None, orig_code = None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):

        # This is perfect options, also iteratively

        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        mask0 = _mask.clone()
        init_code0 = init_code.clone()
        non_empty_indices = [i for i, arr in enumerate(options) if len(arr) > 0]
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code.view(nb_sample, self.patch_size*self.patch_size)
                mask = _mask.view(nb_sample, self.patch_size*self.patch_size)
                # mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(self.args.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.args.device)

            if len(non_empty_indices) != 0:
                if step == 1:
                    with torch.amp.autocast('cuda'):  # half precision
                        logit = self.vit(code.clone().view(nb_sample, self.patch_size, self.patch_size), labels,
                                             drop_label=drop)

                    prob = torch.softmax(logit * sm_temp, -1)

                    # 获取各选项的概率值和最大索引
                    option_tensors = [torch.tensor(options[idx]).to(prob.device) for idx in non_empty_indices]
                    option_probs = [prob[:, idx, opt] for idx, opt in zip(non_empty_indices, option_tensors)]

                    # 获取每个选项中概率最大的值和对应的索引
                    max_vals, max_inds = zip(*(opt_probs.max(dim=1) for opt_probs in option_probs))

                    # 将最大值和最大索引堆叠成张量
                    prob_max_values = torch.stack(max_vals, dim=1)
                    prob_max_indices = torch.stack(max_inds, dim=1)

                    # 使用非空索引的映射，找到实际的最大索引
                    max_indices = torch.stack([opt[idx] for opt, idx in zip(option_tensors, prob_max_indices.t())], dim=1)

                    # 使用掩码扩展并过滤无关选项的概率
                    mask_expanded = mask[:, non_empty_indices]
                    prob_max_values *= mask_expanded
                    # code[:, non_empty_indices][mask_expanded == 1] = max_indices[mask_expanded == 1]
                    # 展平掩码和索引，避免循环赋值
                    flat_indices = mask_expanded.bool().flatten()
                    flattened_code = code[:, non_empty_indices].flatten()
                    flattened_code[flat_indices] = max_indices.flatten()[flat_indices]
                    code[:, non_empty_indices] = flattened_code.view_as(code[:, non_empty_indices])
                    # # Placeholder lists for max values and indices
                    # prob_max_values = []
                    # prob_max_indices = []
                    #
                    # # Loop through each non-empty index to handle variable-length options
                    # for idx in non_empty_indices:
                    #     # Get probability values corresponding to options[idx] for all users
                    #     option_probs = prob[:, idx, torch.tensor(options[idx])]
                    #
                    #     # Find the maximum probability and index within this option set for each user
                    #     max_vals, max_inds = option_probs.max(dim=1)
                    #     prob_max_values.append(max_vals)
                    #     prob_max_indices.append(max_inds)
                    #
                    # # Stack the lists to form tensors
                    # prob_max_values = torch.stack(prob_max_values, dim=1)
                    # prob_max_indices = torch.stack(prob_max_indices, dim=1)
                    # max_indices = torch.empty_like(prob_max_indices)
                    # for idx in range(len(non_empty_indices)):
                    #     new_idx = non_empty_indices[idx]
                    #     max_indices[:,idx] = torch.tensor(options[new_idx]).to(prob.device)[prob_max_indices[:, idx]]
                    #
                    # # Mask probabilities that aren't relevant for each user
                    # mask_expanded = mask[:, non_empty_indices]  # Expand the mask for non-empty indices
                    # prob_max_values *= mask_expanded  # Zero out irrelevant probabilities
                    # code[:, non_empty_indices][mask_expanded == 1] = max_indices[mask_expanded == 1]

                else:
                    # step = len(non_empty_indices)
                    # Instantiate scheduler
                    if isinstance(sched_mode, str):  # Standard ones
                        scheduler = self.adap_sche(step, mode=sched_mode)
                    else:  # Custom one
                        scheduler = sched_mode


                    # Beginning of sampling, t = number of token to predict a step "indice"
                    for indice, t in enumerate(scheduler):
                        if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                            t = int(mask.sum().item())

                        if mask.sum() == 0:  # Break if code is fully predicted
                            break

                        w = 0
                        # with torch.cuda.amp.autocast():  # half precision
                        with torch.amp.autocast('cuda'):  # half precision
                            if w != 0:
                                # Model Prediction
                                logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                                 torch.cat([labels, labels], dim=0),
                                                 torch.cat([~drop, drop], dim=0))
                                # logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                                # _w = w * (indice / (len(scheduler)-1))
                                # # Classifier Free Guidance
                                # logit = (1 + _w) * logit_c - _w * logit_u
                            else:
                                # logit = self.vit(code.clone(), labels, drop_label=~drop)
                                # logit2 = self.vit(code.clone(), labels, drop_label=drop)
                                logit = self.vit(code.clone().view(nb_sample, self.patch_size,self.patch_size), labels, drop_label=drop)
                        # Z0 = options


                        prob = torch.softmax(logit * sm_temp, -1)

                        # 将 `options` 转换成在 `non_empty_indices` 中对应的张量列表
                        option_tensors = [torch.tensor(options[idx], device=prob.device) for idx in non_empty_indices]

                        # 在每个选项集中获取最大概率及其索引
                        option_probs = [prob[:, idx, opt] for idx, opt in zip(non_empty_indices, option_tensors)]
                        max_vals, max_inds = zip(*(opt.max(dim=1) for opt in option_probs))

                        # 将最大值和最大索引堆叠成张量
                        prob_max_values = torch.stack(max_vals, dim=1)
                        prob_max_indices = torch.stack(max_inds, dim=1)

                        # 使用掩码扩展并过滤无关选项的概率
                        mask_expanded = mask[:, non_empty_indices]
                        prob_max_values *= mask_expanded

                        # 获取每个用户中非零掩码的最大概率值及其索引
                        max_conf_values, max_conf_indices = prob_max_values.max(dim=1)

                        # 使用 `non_empty_indices` 的映射找到实际的最大索引
                        IDX = torch.tensor(non_empty_indices, device=prob.device)[max_conf_indices]

                        # 找到有效用户
                        valid_users = mask.sum(dim=1) > 0
                        valid_idx = torch.arange(nb_sample, device=prob.device)[valid_users]
                        valid_IDX = IDX[valid_users]
                        valid_max_conf_indices = max_conf_indices[valid_users]

                        # 获取有效用户的最终选择
                        selected_options = torch.stack(
                            [opt[prob_max_indices[:, i]] for i, opt in enumerate(option_tensors)], dim=1)
                        code[valid_idx, valid_IDX] = selected_options.gather(1, valid_max_conf_indices.view(-1, 1)).squeeze(
                            1)

                        # 仅更新有效用户的掩码
                        mask[valid_idx, valid_IDX] = 0

                        # # # # select masked elements
                        # # for idx in non_empty_indices:
                        # #     conf = prob[:, idx, options[idx]]
                        #
                        # # max_values, pred_code = torch.max(prob, dim=2)
                        # # conf = max_values
                        # # conf[~mask.bool()] = -math.inf
                        # # # chose the predicted token with the highest confidence
                        # # tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                        # # tresh_conf = tresh_conf[:, -1]
                        #
                        # # Placeholder lists for max values and indices
                        # prob_max_values = []
                        # prob_max_indices = []
                        #
                        # # Loop through each non-empty index to handle variable-length options
                        # for idx in non_empty_indices:
                        #     # Get probability values corresponding to options[idx] for all users
                        #     option_probs = prob[:, idx, torch.tensor(options[idx])]
                        #
                        #     # Find the maximum probability and index within this option set for each user
                        #     max_vals, max_inds = option_probs.max(dim=1)
                        #     prob_max_values.append(max_vals)
                        #     prob_max_indices.append(max_inds)
                        #
                        # # Stack the lists to form tensors
                        # prob_max_values = torch.stack(prob_max_values, dim=1)
                        # prob_max_indices = torch.stack(prob_max_indices, dim=1)
                        #
                        # # Mask probabilities that aren't relevant for each user
                        # mask_expanded = mask[:, non_empty_indices]  # Expand the mask for non-empty indices
                        # prob_max_values *= mask_expanded  # Zero out irrelevant probabilities
                        #
                        # # Find the highest probability value for each user, ignoring masked elements
                        # max_conf_values, max_conf_indices = prob_max_values.max(dim=1)
                        #
                        # # Map back to the full index in `code`
                        # IDX = torch.tensor(non_empty_indices, device=prob.device)[max_conf_indices]
                        #
                        # # Identify users with non-zero masks
                        # valid_users = mask.sum(dim=1) > 0
                        #
                        # # Filter only valid users
                        # valid_idx = torch.arange(nb_sample, device=prob.device)[valid_users]
                        # valid_IDX = IDX[valid_users]
                        # valid_max_conf_indices = max_conf_indices[valid_users]
                        #
                        # # Stack tensors and gather only the required values for valid users
                        # code[valid_idx, valid_IDX] = torch.stack(
                        #     [torch.tensor(options[non_empty_indices[i]], device=prob.device)[prob_max_indices[:, i]]
                        #      for i in range(len(non_empty_indices))]
                        #     , dim=1).gather(1, valid_max_conf_indices.view(-1, 1)).squeeze(1)
                        #
                        # # Update mask only for valid users
                        # mask[valid_idx, valid_IDX] = 0

                    # # Update `code` and `mask` for the selected indices
                    # code[torch.arange(nb_sample), IDX] = torch.stack(
                    #     [torch.tensor(options[non_empty_indices[i]], device=prob.device)[prob_max_indices[:, i]]
                    #      for i in range(len(non_empty_indices))]
                    #     , dim=1).gather(1, max_conf_indices.view(-1, 1)).squeeze(1)
                    # mask[torch.arange(nb_sample), IDX] = 0


                    # for userID in range(nb_sample):
                    #     mask_small = mask[userID, non_empty_indices]
                    #     if torch.sum(mask_small)==0:
                    #         continue
                    #     prob_max_values = [
                    #         torch.max(prob[userID, idx, options[idx]], dim=0)[0]  # 直接获取 (max_value, index) 元组
                    #         for idx in non_empty_indices
                    #     ]
                    #     prob_max_indices = [
                    #         torch.max(prob[userID, idx, options[idx]], dim=0)[1]  # 直接获取 (max_value, index) 元组
                    #         for idx in non_empty_indices
                    #     ]
                    #     # indices = torch.nonzero(mask[userID, :] == 1, as_tuple=True)
                    #
                    #     prob_max_values = torch.tensor(prob_max_values) * mask_small.cpu()
                    #     val, idx = torch.max(prob_max_values, dim=0)
                    #
                    #     # Get the index in 0-255
                    #     IDX = non_empty_indices[idx]
                    #     # update the value of this user
                    #     code[userID, IDX] = options[IDX][prob_max_indices[idx]]
                    #     # update the mask of this user
                    #     mask[userID, IDX] = 0

                    # # replace the chosen tokens
                    # conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                    # f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                    # code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]
                    #
                    # # update the mask
                    # for i_mask, ind_mask in enumerate(indice_mask):
                    #     mask[i_mask, ind_mask] = 0
                    # l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                    # l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())
            else:
                print('No collisions')


            # decode the final prediction
            _code = torch.clamp(code.view(nb_sample, self.patch_size,self.patch_size), 0,  self.codebook_size-1)

            init_code0[mask0 == 1] = torch.randint(0, 1024, (torch.sum(mask0 == 1),), dtype=torch.int64).to(_code.device)

            # init_code0 = init_code0.view(nb_sample, self.patch_size * self.patch_size)
            # mask0 = mask0.view(nb_sample, self.patch_size * self.patch_size)
            # # Iterate over non_empty_indices only once and minimize operations inside the loop
            # for idx in non_empty_indices:
            #     # Select options once and move it to the device only once
            #     selected_options = torch.tensor(options[idx], device=init_code0.device)
            #
            #     # Use boolean indexing to get the shape for random sampling
            #     mask_indices = mask0[:, idx].nonzero(as_tuple=True)[0]
            #     if mask_indices.numel() > 0:  # Skip if there are no valid indices
            #         # Randomly select indices, but match dimensions by expanding the selection
            #         random_indices = torch.randint(
            #             0, len(selected_options), size=(mask_indices.numel(),), device=init_code0.device
            #         )
            #
            #         # Get the random selections
            #         random_choices = selected_options[random_indices]
            #
            #         # Assign the random choices directly
            #         init_code0[mask_indices, idx] = random_choices
            # init_code0 = init_code0.view(nb_sample, self.patch_size ,self.patch_size)
            # mask0 = mask0.view(nb_sample, self.patch_size, self.patch_size)


            Ocode = orig_code.view(nb_sample, self.patch_size * self.patch_size)
            mask0 = mask0.view(nb_sample, self.patch_size*self.patch_size)
            # debug2 = Ocode[mask0 == 1]
            _code = _code.view(nb_sample, self.patch_size * self.patch_size)
            init_code0 = init_code0.view(nb_sample, self.patch_size * self.patch_size)


            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=_code.shape[0], init=_code.cpu().numpy(), n_init=1, random_state=0)
            kmeans.fit(Ocode.cpu().numpy())
            labels = kmeans.labels_
            _code_reorder = _code[labels]
            init_code0_reorder = init_code0[labels]
            mask0_reorder = mask0[labels]
            debug2 = Ocode[mask0_reorder == 1]
            N_err = debug2.size()[0]
            debug1 = _code_reorder[mask0_reorder == 1]
            Correct_Rate = torch.sum(debug1-debug2 == 0)/N_err
            debug3 = init_code0_reorder[mask0_reorder == 1]
            Correct_Rate2 = torch.sum(debug3 - debug2 == 0) / N_err


            x = self.ae.decode_code(_code_reorder.view(nb_sample, self.patch_size,self.patch_size))
            x_orig = self.ae.decode_code(orig_code.view(nb_sample, self.patch_size,self.patch_size))
            x_random = self.ae.decode_code(init_code0_reorder.view(nb_sample, self.patch_size,self.patch_size))

        mask_final_1 = orig_code - _code_reorder.view(nb_sample, self.patch_size, self.patch_size)
        mask_final_1[mask_final_1 != 0] = 1
        mask_final_2 = orig_code - init_code0_reorder.view(nb_sample, self.patch_size, self.patch_size)
        mask_final_2[mask_final_2 != 0] = 1
        # self.vit.train()
        return x, x_orig, x_random, mask_final_1, mask_final_2, Correct_Rate, Correct_Rate2, N_err
        # return x, x_orig, x_random, l_codes, l_mask, Correct_Rate, Correct_Rate2, N_err

    ############################ Hi, I solved the problem ################################
    def GenDecoding_iterative(self, init_code=None, options=None, _mask=None, orig_code=None, nb_sample=50,
                            labels=None, sm_temp=1, w=3,
                            randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):

        # This is perfect options, also iteratively

        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        mask0 = _mask.clone()
        init_code0 = init_code.clone()
        non_empty_indices = [i for i, arr in enumerate(options) if len(arr) > 0]
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []  # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                if nb_sample<10:
                    if nb_sample == 2:
                        labels = [282, 7]
                    if nb_sample == 3:
                        labels = [282, 7, 1]
                    if nb_sample == 4:
                        labels = [282, 7, 1, 604]
                    if nb_sample == 5:
                        labels = [282, 7, 1, 850, 604]
                    # labels = [1] * nb_sample
                    labels = torch.LongTensor(labels).to(self.args.device)
                else:
                    labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                    labels = torch.LongTensor(labels).to(self.args.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.args.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code.view(nb_sample, self.patch_size * self.patch_size)
                mask = _mask.view(nb_sample, self.patch_size * self.patch_size)
                # mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size)
            else:  # Initialize a code
                if self.args.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, self.codebook_size, (nb_sample, self.patch_size, self.patch_size)).to(
                        self.args.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.args.mask_value).to(
                        self.args.device)
                mask = torch.ones(nb_sample, self.patch_size * self.patch_size).to(self.args.device)


            # step = len(non_empty_indices)
            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            stored_codes = []
            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                w = 0
                # with torch.cuda.amp.autocast():  # half precision
                with torch.amp.autocast('cuda'):  # half precision
                    if w != 0:
                        # Model Prediction
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0))
                        # logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        # _w = w * (indice / (len(scheduler)-1))
                        # # Classifier Free Guidance
                        # logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        # logit = self.vit(code.clone(), labels, drop_label=~drop)
                        # logit2 = self.vit(code.clone(), labels, drop_label=drop)
                        # logit = self.vit(code.clone().view(nb_sample, self.patch_size, self.patch_size),
                        #                  labels, drop_label=drop)
                        logit = self.vit(code.clone().view(nb_sample, self.patch_size, self.patch_size),
                                         labels, drop_label=~drop)
                # Z0 = options

                prob = torch.softmax(logit * sm_temp, -1)

                non_empty_indices = [i for i, arr in enumerate(options) if len(arr) > 0]

                # # Without options
                # mask_expanded = mask.unsqueeze(2).expand(nb_sample, 256, 1025)
                # prob *= mask_expanded
                # max_idx_1024 = torch.argmax(prob, dim=2)
                # max_vals_1024 = torch.gather(prob, 2, max_idx_1024.unsqueeze(2)).squeeze(2)
                # max_vals_256, max_idx_256 = torch.max(max_vals_1024, dim=1)  # 最大值和对应索引
                # code[torch.arange(0, nb_sample), max_idx_256] = max_idx_1024[torch.arange(0, nb_sample), max_idx_256]
                # mask[torch.arange(0, nb_sample), max_idx_256] = 0

                # options_unique = []
                # for array in options:
                #     unique_elements, counts = np.unique(array, return_counts=True)  # 找到唯一值和它们的出现次数
                #     non_repeated = unique_elements[counts == 1]  # 仅保留出现次数为 1 的元素
                #     options_unique.append(non_repeated)

                # 将 `options` 转换成在 `non_empty_indices` 中对应的张量列表
                # option_tensors = [torch.tensor(options_unique[idx], device=prob.device) for idx in
                #                   non_empty_indices]
                option_tensors = [torch.tensor(options[idx], device=prob.device) for idx in
                                  non_empty_indices]

                # 在每个选项集中获取最大概率及其索引
                option_probs = [prob[:, idx, opt] for idx, opt in zip(non_empty_indices, option_tensors)]
                max_vals, max_inds = zip(*(opt.max(dim=1) for opt in option_probs))


                # 将最大值和最大索引堆叠成张量
                prob_max_values = torch.stack(max_vals, dim=1)
                prob_max_indices = torch.stack(max_inds, dim=1)

                # # AAA = np.array(options).T
                # U1 = prob_max_indices[0,:]
                # xvector1 = np.array([options[i9][U1[i9]] for i9 in range(len(U1))])
                # U2 = prob_max_indices[1, :]
                # xvector2 = np.array([options[i9][U2[i9]] for i9 in range(len(U2))])
                # tensor1 = torch.from_numpy(xvector1).to(init_code.device)
                # tensor2 = torch.from_numpy(xvector2).to(init_code.device)
                # stacked_tensor = torch.stack((tensor1, tensor2), dim=0)
                # ImgRecon = self.ae.decode_code(stacked_tensor.view(nb_sample, self.patch_size, self.patch_size))

                # 使用掩码扩展并过滤无关选项的概率
                mask_expanded = mask[:, non_empty_indices]
                prob_max_values *= mask_expanded

                # if nb_sample == 2:
                #     # 获取每个用户中非零掩码的最大概率值及其索引
                #     prob_max_sum = torch.sum(prob_max_values, axis=0)
                #     max_value, max_index = torch.max(prob_max_sum, dim=0)
                #     tokk = options[max_index]
                #     idxx = prob_max_indices[:, max_index]
                #     if idxx[0]==idxx[1]:
                #         vals = prob_max_values[:,max_index]
                #         ma_val, ma_ind = torch.max(vals, dim=0)
                #         code[ma_ind, max_index] = tokk[prob_max_indices[ma_ind, max_index]]
                #         code[1-ma_ind, max_index] = tokk[1-prob_max_indices[ma_ind, max_index]]
                #     else:
                #         code[0, max_index] = tokk[idxx[0]]
                #         code[1, max_index] = tokk[idxx[1]]
                #     mask[:, max_index] = 0
                # else:
                #     prob_max_sum = torch.sum(prob_max_values, axis=0)/(torch.sum(mask, dim=0) + 1e-5)
                #     max_value, max_index = torch.max(prob_max_sum, dim=0)
                #     tokk = options[max_index]
                #     mask_col = mask[:,max_index]
                #     idxx = prob_max_indices[mask_col == 1, max_index]
                #     if idxx.numel() != torch.unique(idxx).numel():
                #         vals = prob_max_values[:, max_index]
                #         ma_val, ma_ind = torch.max(vals, dim=0)
                #         code[ma_ind, max_index] = tokk[prob_max_indices[ma_ind, max_index]]
                #         # code[1 - ma_ind, max_index] = tokk[1 - prob_max_indices[ma_ind, max_index]]
                #         mask[ma_ind, max_index] = 0
                #     else:
                #         indiii = torch.where(mask_col == 1)[0]
                #         for ii in range(len(idxx)):
                #             # if mask_col[ii] == 1:
                #             code[indiii[ii], max_index] = tokk[idxx[ii]]
                #         # code[1, max_index] = tokk[idxx[1]]
                #         mask[:, max_index] = 0

                # prob_max_sum = torch.sum(prob_max_values, axis=0) / (torch.sum(mask[:, non_empty_indices], dim=0) + 1e-5)
                # max_conf_values, max_conf_indices = torch.topk(prob_max_sum, k=nb_sample, dim=0)
                # valid_idx = torch.argmax(prob_max_values[:, max_conf_indices], dim=0)


                max_conf_values, max_conf_indices = prob_max_values.max(dim=1)

                # 使用 `non_empty_indices` 的映射找到实际的最大索引
                IDX = torch.tensor(non_empty_indices, device=prob.device)[max_conf_indices]

                # 找到有效用户
                valid_users = mask.sum(dim=1) > 0
                valid_idx = torch.arange(nb_sample, device=prob.device)[valid_users]
                valid_IDX = IDX[valid_users]
                valid_max_conf_indices = max_conf_indices[valid_users]

                # 获取有效用户的最终选择
                selected_options = torch.stack(
                    [opt[prob_max_indices[:, i]] for i, opt in enumerate(option_tensors)], dim=1)
                code[valid_idx, valid_IDX] = selected_options.gather(1, valid_max_conf_indices.view(-1,
                                                                                                    1)).squeeze(
                    1)
                # selected_options_r = torch.stack(
                #     [opt[1-prob_max_indices[:, i]] for i, opt in enumerate(option_tensors)], dim=1)
                # code[1-valid_idx, valid_IDX] = selected_options_r.gather(1, valid_max_conf_indices.view(-1,1)).squeeze(1)
                # # 仅更新有效用户的掩码
                # mask[:, valid_IDX] = 0
                mask[valid_idx, valid_IDX] = 0

                for iii in range(len(valid_IDX)):
                    codeexpand = code[iii, valid_IDX[iii]].cpu().numpy()
                    mask00 = options[valid_IDX[iii]] != codeexpand  # 创建布尔掩码
                    # options[valid_IDX[iii]] = options[valid_IDX[iii]][mask00]  # 更新 options
                    false_count = np.sum(mask00 == False)
                    if len(options[valid_IDX[iii]]) >1 and false_count > 1:
                        false_indices = np.where(mask00 == False)[0]
                        keep_index = np.random.choice(false_indices)
                        # 将其他 False 修改为 True
                        mask00[false_indices] = True
                        mask00[keep_index] = False
                    if not mask00.any():
                        options[valid_IDX[iii]] = np.array([])
                    else:
                        options[valid_IDX[iii]] = options[valid_IDX[iii]][mask00]

                if indice%20 == 0:
                    codesave = code.clone()
                    # for i in range(256):
                    #     if mask[0, i] == 1:
                    #         # 随机打乱 options[:, i] 的元素顺序
                    #         # shuffled_elements = np.random.permutation(options[i])
                    #         shuffled_elements = np.random.permutation(options_unique[i])
                    #         codesave[:, i] = torch.tensor(shuffled_elements).to(self.args.device)
                    for jjjj in range(nb_sample):
                        for iiii in range(256):
                            if mask[jjjj, iiii] == 1 and len(options[iiii]) != 0:
                                # codesave[j, i] = torch.tensor(np.random.choice(options_unique[i])).to(self.args.device)
                                codesave[jjjj, iiii] = torch.tensor(np.random.choice(options[iiii])).to(self.args.device)
                    stored_codes.append(codesave.clone())

            saved_Imgs = []
            for idx, stored_code in enumerate(stored_codes):
                _stored_code = torch.clamp(stored_code.view(nb_sample, self.patch_size, self.patch_size), 0, self.codebook_size - 1)
                ImgR1 = self.ae.decode_code(_stored_code)
                saved_Imgs.append(ImgR1.clone())

            to_pil = ToPILImage()
            num_images = len(saved_Imgs)  # 图片数量
            images_per_row = num_images  # 每行图片数量（默认全部在一行）
            num_rows = nb_sample  # 每个通道占一行

            # 创建绘图
            fig, axs = plt.subplots(num_rows, images_per_row,
                                    figsize=(12, num_rows),
                                    gridspec_kw={'wspace': 0.001, 'hspace': 0.1})  # 调整行间距
            axs = axs.reshape(num_rows, -1)  # 将轴重新组织为 (通道数, 图片数/行)

            # 遍历通道和图片
            for channel_idx in range(nb_sample):
                for idx, imggg in enumerate(saved_Imgs):
                    imggg = torch.clamp(imggg, 0, 1)  # 确保值在 [0, 1] 范围内
                    img_pil = to_pil(imggg[channel_idx, :])  # 转为 PIL 图像
                    axs[channel_idx, idx].imshow(img_pil)
                    axs[channel_idx, idx].axis('off')  # 去除坐标轴

            plt.tight_layout()
            plt.show()

            # to_pil = ToPILImage()
            # # 确定要显示的图片数和每行图片数量
            # num_images = len(saved_Imgs)
            # images_per_row = num_images  # 每行显示的图片数量
            # num_rows = 1
            # # 创建绘图
            # # fig, axs = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows))
            # # 创建绘图，减少 wspace 和 hspace
            # fig, axs = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows), gridspec_kw={'wspace': 0.01, 'hspace': 0.2})
            # axs = axs.flatten()  # 将轴展平成一维列表，方便索引
            # # 遍历图片并绘制
            # for idx, imggg in enumerate(saved_Imgs):
            #     imggg = torch.clamp(imggg, 0, 1)  # 确保值在 [0, 1] 范围内
            #     img_pil = to_pil(imggg[0,:])  # 转为 PIL 图像
            #     axs[idx].imshow(img_pil)
            #     axs[idx].axis('off')  # 去除坐标轴
            #     # axs[idx].set_title(f"Image {idx + 1}")
            # plt.tight_layout()
            # # plt.tight_layout(pad=0.5)
            # plt.show()
            #
            # fig, axs = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows),
            #                         gridspec_kw={'wspace': 0.01, 'hspace': 0.2})
            # axs = axs.flatten()  # 将轴展平成一维列表，方便索引
            # # 遍历图片并绘制
            # for idx, imggg in enumerate(saved_Imgs):
            #     imggg = torch.clamp(imggg, 0, 1)  # 确保值在 [0, 1] 范围内
            #     img_pil = to_pil(imggg[1, :])  # 转为 PIL 图像
            #     axs[idx].imshow(img_pil)
            #     axs[idx].axis('off')  # 去除坐标轴
            #     # axs[idx].set_title(f"Image {idx + 1}")
            # plt.tight_layout()
            # # plt.tight_layout(pad=0.5)
            # plt.show()
            #
            # if nb_sample > 2:
            #     fig, axs = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows),
            #                             gridspec_kw={'wspace': 0.01, 'hspace': 0.2})
            #     axs = axs.flatten()  # 将轴展平成一维列表，方便索引
            #     # 遍历图片并绘制
            #     for idx, imggg in enumerate(saved_Imgs):
            #         imggg = torch.clamp(imggg, 0, 1)  # 确保值在 [0, 1] 范围内
            #         img_pil = to_pil(imggg[2, :])  # 转为 PIL 图像
            #         axs[idx].imshow(img_pil)
            #         axs[idx].axis('off')  # 去除坐标轴
            #         # axs[idx].set_title(f"Image {idx + 1}")
            #     plt.tight_layout()
            #     # plt.tight_layout(pad=0.5)
            #     plt.show()
            code = torch.clamp(code.view(nb_sample, self.patch_size, self.patch_size), 0,
                                       self.codebook_size - 1)
            ImgRecon = self.ae.decode_code(code.view(nb_sample, self.patch_size, self.patch_size))

            to_pil = ToPILImage()
            # 假设 ImgRecon 是 4D 张量 [nb_sample, C, H, W]
            ImgRecon = torch.clamp(ImgRecon, 0, 1)  # 确保值在 [0, 1] 范围内

            for i in range(ImgRecon.size(0)):
                img_pil = to_pil(ImgRecon[i])  # 转为 PIL 图像
                plt.figure()
                plt.imshow(img_pil)
                plt.axis('off')
                plt.title(f"Image {i + 1}")
                plt.show()

            # decode the final prediction
            _code = torch.clamp(code.view(nb_sample, self.patch_size, self.patch_size), 0, self.codebook_size - 1)

            init_code0[mask0 == 1] = torch.randint(0, 1024, (torch.sum(mask0 == 1),), dtype=torch.int64).to(
                _code.device)

            # init_code0 = init_code0.view(nb_sample, self.patch_size * self.patch_size)
            # mask0 = mask0.view(nb_sample, self.patch_size * self.patch_size)
            # # Iterate over non_empty_indices only once and minimize operations inside the loop
            # for idx in non_empty_indices:
            #     # Select options once and move it to the device only once
            #     selected_options = torch.tensor(options[idx], device=init_code0.device)
            #
            #     # Use boolean indexing to get the shape for random sampling
            #     mask_indices = mask0[:, idx].nonzero(as_tuple=True)[0]
            #     if mask_indices.numel() > 0:  # Skip if there are no valid indices
            #         # Randomly select indices, but match dimensions by expanding the selection
            #         random_indices = torch.randint(
            #             0, len(selected_options), size=(mask_indices.numel(),), device=init_code0.device
            #         )
            #
            #         # Get the random selections
            #         random_choices = selected_options[random_indices]
            #
            #         # Assign the random choices directly
            #         init_code0[mask_indices, idx] = random_choices
            # init_code0 = init_code0.view(nb_sample, self.patch_size ,self.patch_size)
            # mask0 = mask0.view(nb_sample, self.patch_size, self.patch_size)

            Ocode = orig_code.view(nb_sample, self.patch_size * self.patch_size)
            mask0 = mask0.view(nb_sample, self.patch_size * self.patch_size)
            # debug2 = Ocode[mask0 == 1]
            _code = _code.view(nb_sample, self.patch_size * self.patch_size)
            init_code0 = init_code0.view(nb_sample, self.patch_size * self.patch_size)

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=_code.shape[0], init=_code.cpu().numpy(), n_init=1, random_state=0)
            kmeans.fit(Ocode.cpu().numpy())
            labels = kmeans.labels_
            _code_reorder = _code[labels]
            init_code0_reorder = init_code0[labels]
            mask0_reorder = mask0[labels]
            debug2 = Ocode[mask0_reorder == 1]
            N_err = debug2.size()[0]
            debug1 = _code_reorder[mask0_reorder == 1]
            Correct_Rate = torch.sum(debug1 - debug2 == 0) / N_err
            debug3 = init_code0_reorder[mask0_reorder == 1]
            Correct_Rate2 = torch.sum(debug3 - debug2 == 0) / N_err

            x = self.ae.decode_code(_code_reorder.view(nb_sample, self.patch_size, self.patch_size))
            x_orig = self.ae.decode_code(orig_code.view(nb_sample, self.patch_size, self.patch_size))
            x_random = self.ae.decode_code(init_code0_reorder.view(nb_sample, self.patch_size, self.patch_size))

        # self.vit.train()
        return x, x_orig, x_random, l_codes, l_mask, Correct_Rate, Correct_Rate2, N_err