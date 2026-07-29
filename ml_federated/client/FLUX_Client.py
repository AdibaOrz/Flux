# ml_federated/client/FLUX_Client.py

import copy
import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from ml_federated.client.Base_Client import Base_Client


class FLUX_Client(Base_Client):
    def __init__(self, client_id, dataset_configs, device, args):
        super().__init__(client_id, dataset_configs, device, args)
        self.args = args
        self.num_classes = dataset_configs['num_classes']

        self.p_head = None

        self.best_full_state = None
        self.best_p_head_state = None
        self.best_val_f1 = -1.0

    def local_train(self, passed_model):
        model = copy.deepcopy(passed_model).to(self.device)
        self._ensure_p_head(model)
        self._set_model_mode(model, mode='train')

        opt_shared, opt_personal = self._get_optimizers(model)
        ce = self.criterion

        epoch_losses = []
        epoch_metas = []
        for _ in range(self.local_epochs):
            shared_losses, private_losses = [], []
            batch_metas = []
            for x, mask, y in self.train_loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()

                opt_shared.zero_grad()
                logits_g, unimodal_logits, fusion_out, unimodal_logvars, mod_mask, valid_mask = model(x, mask)
                y = y[valid_mask]
                loss_task = ce(logits_g, y)

                unimodal_losses = []
                uni_metas = {}
                for midx, (u_logits, u_logvars) in enumerate(zip(unimodal_logits, unimodal_logvars)):
                    curr_mod_mask = mod_mask[:, midx]
                    u_loss, u_meta = confidence_tempering_loss('unimodal', u_logits, u_logvars, y, curr_mod_mask)
                    unimodal_losses.append(u_loss)

                    uni_metas[f'uni_m:{midx}_mean_sigma'] = u_meta['uni_mean_sigma']
                    uni_metas[f'uni_m:{midx}_mean_T'] = u_meta['uni_mean_T']

                uni_metas[f'uni_m:ALL_mean_sigma'] = np.mean([v for k, v in uni_metas.items() if ('mean_sigma' in k
                                                                    and k.startswith('uni_m:') and v is not None)])
                uni_metas[f'uni_m:ALL_mean_T'] = np.mean([v for k, v in uni_metas.items() if ('mean_T' in k
                                                                    and k.startswith('uni_m:') and v is not None)])

                loss_confidence_tempering = torch.stack(unimodal_losses).mean()
                loss_shared = loss_task + loss_confidence_tempering
                loss_shared.backward()
                opt_shared.step()

                opt_personal.zero_grad()
                fusion_out_detached = fusion_out.detach()
                unimodal_logvars_detached = unimodal_logvars.detach()
                logits_p = self.p_head(fusion_out_detached)

                mm_loss, mm_meta = confidence_tempering_loss('multimodal',
                                                logits_p, unimodal_logvars_detached, y, mod_mask)
                batch_metas.append({**uni_metas, **mm_meta})

                loss_p = mm_loss
                loss_p.backward()
                opt_personal.step()

                shared_losses.append(loss_shared.item())
                private_losses.append(loss_p.item())

            epoch_shared_loss = float(np.mean(shared_losses)) if shared_losses else math.nan
            epoch_private_loss = float(np.mean(private_losses)) if private_losses else math.nan
            avg = (epoch_shared_loss + epoch_private_loss) / 2.0
            epoch_losses.append(avg)

            epoch_meta_dict = {}
            for key in batch_metas[0].keys():
                values = [bm[key] for bm in batch_metas if bm[key] is not None]
                epoch_meta_dict[key] = float(np.mean(values)) if values else None
            epoch_metas.append(epoch_meta_dict)

        avg_epoch_loss = float(np.mean(epoch_losses))

        avg_epoch_meta = {}
        for key in epoch_metas[0].keys():
            values = [em[key] for em in epoch_metas if em[key] is not None]
            avg_epoch_meta[key] = float(np.mean(values)) if values else None

        return model, avg_epoch_loss, avg_epoch_meta

    @torch.no_grad()
    def local_test(self, model, eval_on):
        self._ensure_p_head(model)

        if eval_on == 'test':
            if self.best_full_state is not None:
                model.load_state_dict(self.best_full_state)
            if self.best_p_head_state is not None:
                self.p_head.load_state_dict(self.best_p_head_state)

        self._set_model_mode(model, mode='eval')
        eval_loader = self.test_loader if eval_on == 'test' else self.val_loader

        y_true_list, y_pred_list = [], []
        for x, mask, y in eval_loader:
            x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device).long()

            logits_g, unimodal_logits, fusion_out, unimodal_logvars, mod_mask, valid_mask = model(x, mask)
            y = y[valid_mask]

            fusion_out_detached = fusion_out.detach()
            logits_p = self.p_head(fusion_out_detached)

            logits_g_detached = logits_g.detach()
            logits = logits_p + logits_g_detached

            _, predicted = torch.max(logits, 1)
            y_true_list = np.concatenate((y_true_list, y.cpu()), axis=0)
            y_pred_list = np.concatenate((y_pred_list, predicted.cpu()), axis=0)

        f1_macro = f1_score(y_true_list, y_pred_list, average='macro')
        if eval_on == 'val':
            if f1_macro > self.best_val_f1:
                self.best_val_f1 = f1_macro
                self.best_full_state = copy.deepcopy(model.state_dict())
                self.best_p_head_state = copy.deepcopy(self.p_head.state_dict())

        return f1_macro

    def _set_model_mode(self, model, mode):
        for component in [model, self.p_head]:
            if component is None:
                continue
            component.to(self.device)
            if mode == 'train':
                component.train()
            else:
                component.eval()

    def _ensure_p_head(self, model):
        if self.p_head is None:
            self.p_head = copy.deepcopy(model.head).to(self.device)

    def _get_optimizers(self, model):
        lr = self.dataset_configs['local_lr']
        wd = self.dataset_configs['local_wd']

        opt_shared = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        personal_params = []
        if self.p_head is not None:
            personal_params += list(self.p_head.parameters())

        opt_personal = torch.optim.SGD(personal_params, lr=lr, weight_decay=wd, momentum=0.9) if personal_params else None
        return opt_shared, opt_personal


def _safe_exp(x, cap=40.0):
    return torch.exp(torch.clamp(x, -cap, cap))


def confidence_tempering_loss(mode, logits, log_var, target, curr_mod_mask, min_temp=1.0, eps=1e-8):
    """Cross-entropy with temperature from per-modality log-variance (confidence tempering)."""
    if mode == 'unimodal':
        uni_meta = {'uni_mean_sigma': None, 'uni_mean_T': None}
        keep_idx = torch.where(curr_mod_mask == 1)[0]
        if keep_idx.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True), uni_meta

        logits = logits[keep_idx]
        log_var = log_var[keep_idx]
        target = target[keep_idx]
        sigma = _safe_exp(0.5 * log_var)
        temperature = sigma + 1e-6
        scaled_logits = logits / temperature

        uni_meta['uni_mean_sigma'] = float(sigma.mean().item())
        uni_meta['uni_mean_T'] = float(temperature.mean().item())
        return F.cross_entropy(scaled_logits, target), uni_meta

    elif mode == 'multimodal':
        avail_mask = curr_mod_mask.transpose(0, 1).float()
        sigma2_m = _safe_exp(log_var).mean(dim=-1)
        precision_m = avail_mask / (sigma2_m + eps)
        precision_sum = precision_m.sum(dim=0)

        sigma2_eff = torch.where(
            precision_sum > 0,
            1.0 / (precision_sum + eps),
            torch.full_like(precision_sum, min_temp ** 2)
        )
        T = torch.sqrt(sigma2_eff) + 1e-6
        scaled_logits = logits / T.unsqueeze(dim=-1)

        mm_meta = {
            'mm_mean_sigma2_eff': float(sigma2_eff.mean().item()),
            'mm_mean_T': float(T.mean().item())
        }
        return F.cross_entropy(scaled_logits, target), mm_meta
