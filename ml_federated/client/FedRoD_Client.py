# ml_federated/client/FedRoD_Client.py

import copy
import math
import torch
import numpy as np
from sklearn.metrics import f1_score
from ml_federated.client.Base_Client import Base_Client


# Equation (6) from FedRoD (https://openreview.net/pdf?id=I1hQbx10Kxn) paper:
# l_bsm = -log(softmax(logits + log_prior[target]))

class BalancedSoftmaxLoss:
    def __init__(self, class_counts, gamma=1.0):
        counts = class_counts.float()
        # log_prior[c] = gamma * log(N_c) if N_c > 0 else -inf
        log_prior = torch.where(
            counts > 0, counts.log() * gamma, # gamma * log(N_c) if N_c > 0
            torch.full_like(counts, float('-inf')), # so that the model is never rewarded for predicting an unseen class.
        )
        self.log_prior = log_prior.view(1, -1)  # [1, C]

    def __call__(self, logits, target, criterion):
        # logits: [N, C]; target: [N]
        # Call BalancedSoftmaxLoss() only in train mode.
        return criterion(logits + self.log_prior.to(logits.device), target)


class FedRoD_Client(Base_Client):
    """
    FedRoD Client for multimodal FL.
    """
    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)
        self.num_classes = dataset_configs['num_classes']

        # Personalized modules (classifier only; fusion stays global)
        self.p_head = None

        # Best checkpoints
        self.best_full_state = None
        self.best_p_head_state = None
        self.best_val_f1 = -1.0

        # Options
        self.brm_mode = self.args.fedrod_brm_mode # 'balanced_softmax' or 'none' # (cross-entropy loss) for none
        self.brm_gamma = 1.0 # Same as in the FedRoD paper

        # Cache
        self._cached_class_hist = None # Cached class counts for balanced softmax loss


    def local_train(self, passed_model):
        # Create a copy to avoid modifying the original model
        model = copy.deepcopy(passed_model).to(self.device)

        # --- Personalized module(s) initialization --- #
        # If it is the first time calling this client's local_train, they will be initialized with global model
        # Otherwise, they will load the latest local state
        self._ensure_p_head(model) # Initialize p_head if it does not exist

        # Sets both the shared (model) and personal module(s) to train mode
        self._set_model_mode(model, mode='train')

        # Get both shared and personal optimizers
        opt_shared, opt_personal = self._get_optimizers(model)
        ce = self.criterion # Cross-entropy loss (ERM) for private params
        brm = self._brm_wrapper() # Balanced softmax loss (BRM) for shared params

        epoch_losses = []
        for _ in range(self.local_epochs):
            shared_losses, private_losses = [], []
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()

                # ---- Shared pass ---- #
                opt_shared.zero_grad()
                logits_g, _, _, fusion_out, valid_mask = model(x, mask, return_for_fedrod=True)
                y = y[valid_mask]  # Filter out invalid samples
                loss_shared = brm(logits_g, y, ce)  # Balanced softmax loss if brm_mode is 'balanced_softmax' else cross-entropy loss
                loss_shared.backward()
                opt_shared.step()

                # ---- Personal pass ---- #
                if opt_personal is not None:
                    opt_personal.zero_grad()
                    # Detach to avoid backprop through shared parameters
                    logits_g_detached = logits_g.detach()
                    fusion_out_detached = fusion_out.detach()

                    feats_priv = fusion_out_detached

                    logits_p = self.p_head(feats_priv) # Personal head
                    loss_p = ce(logits_p + logits_g_detached, y) # Same as in the FedRoD paper (y_pred_shared + y_pred_personal)
                    loss_p.backward()
                    opt_personal.step()
                else:
                    loss_p = torch.tensor(0.0, device=self.device)

                # ---- Log losses ---- #
                shared_losses.append(loss_shared.item())
                private_losses.append(loss_p.item())
            epoch_shared_loss = float(np.mean(shared_losses)) if shared_losses else math.nan
            epoch_private_loss = float(np.mean(private_losses)) if private_losses else math.nan
            avg = (epoch_shared_loss + epoch_private_loss) / 2.0 # Average loss across shared and personal losses
            epoch_losses.append(avg)

        avg_epoch_loss = float(np.mean(epoch_losses))

        # Stash new private params for the next round
        return model, avg_epoch_loss


    @torch.no_grad()
    def local_test(self, model, eval_on):
        # --- Personalized module(s) initialization --- #
        # If it is the first time calling this client's local_test, they will be initialized with global model
        # Otherwise, they will load the latest local state
        self._ensure_p_head(model)

        if eval_on == 'test': # If test, load the best local snapshot
            if self.best_full_state is not None:
                model.load_state_dict(self.best_full_state) # includes shared + private params
            if self.best_p_head_state is not None:
                self.p_head.load_state_dict(self.best_p_head_state)

        self._set_model_mode(model, mode='eval')
        eval_loader = self.test_loader if eval_on == 'test' else self.val_loader

        y_true_list, y_pred_list = [], []
        for x, mask, y in eval_loader:
            x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()

            # ---- Shared pass ---- #
            logits_g, _, _, fusion_out, valid_mask = model(x, mask, return_for_fedrod=True)
            y = y[valid_mask]  # Filter out invalid samples

            # ---- Personal pass ---- #
            logits_g_detached = logits_g.detach()
            fusion_out_detached = fusion_out.detach()

            feats_priv = fusion_out_detached

            logits_p = self.p_head(feats_priv)
            logits = logits_p + logits_g_detached  # Same as in the FedRoD paper
            _, predicted = torch.max(logits, 1)
            y_true_list = np.concatenate((y_true_list, y.cpu()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
        if eval_on == 'val': # Get the snapshot of the best model
            if f1_macro > self.best_val_f1:
                # print(f'New best validation F1 score for client {self.client_id} has been updated. New best val F1 score is: ', f1_macro)
                self.best_val_f1 = f1_macro
                self.best_full_state = copy.deepcopy(model.state_dict())
                self.best_p_head_state = copy.deepcopy(self.p_head.state_dict())
        return f1_macro


    def _set_model_mode(self, model, mode):
        for component in [model, self.p_head]:
            if component is None: continue
            component.to(self.device)
            if mode == 'train':
                component.train()
            else:
                component.eval()


    # ====== Fed-RoD specific helpers ====== #
    def _ensure_p_head(self, model):
        if self.p_head is None: # The first time p_head is initialized
            self.p_head = copy.deepcopy(model.head).to(self.device)


    def _class_histogram(self):
        if self._cached_class_hist is not None:
            return self._cached_class_hist
        # Count the number of samples in each class in training set (to use for balanced softmax loss)
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        if self.train_loader is not None:
            for _, _, y in self.train_loader:
                y = y.view(-1).long() # Ensure y is a 1D tensor
                binc = torch.bincount(y, minlength=self.num_classes)
                counts += binc
        self._cached_class_hist = counts
        return self._cached_class_hist


    def _brm_wrapper(self):
        if self.brm_mode == 'balanced_softmax':
            return BalancedSoftmaxLoss(self._class_histogram(), gamma=self.brm_gamma)
        elif self.brm_mode == 'none': # Use cross-entropy loss
            return lambda logits, target, criterion: criterion(logits, target)
        else:
            raise ValueError(f"Invalid BRM mode: {self.brm_mode}. Choose 'balanced_softmax' or 'none'.")


    def _get_optimizers(self, model):
        lr = self.dataset_configs['local_lr']
        wd = self.dataset_configs['local_wd']

        # Optimizer for shared parameters (typical approach like in FedAvg)
        opt_shared = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

        # Optimizer for personal parameters (head)
        personal_params = []
        if self.p_head is not None:
            personal_params += list(self.p_head.parameters())
        opt_personal = torch.optim.SGD(personal_params, lr=lr, weight_decay=wd, momentum=0.9) if personal_params else None

        return opt_shared, opt_personal
