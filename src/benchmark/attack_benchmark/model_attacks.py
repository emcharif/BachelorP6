import copy
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch_geometric.loader import DataLoader


class model_attacks:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def _get_model_device(self, model):
        return next(model.parameters()).device

    def _move_batch_to_device(self, batch, device):
        if hasattr(batch, "to"):
            return batch.to(device)
        return batch

    def blind_fine_tune_attack(
        self,
        model,
        attacker_dataset,
        enable_prints=False,
        epochs=10,
        learning_rate=0.001,
    ):
        attacked_model = copy.deepcopy(model)
        device = self._get_model_device(attacked_model)

        loader = DataLoader(attacker_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(attacked_model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            attacked_model.train()
            total_loss = 0.0

            for batch in loader:
                batch = self._move_batch_to_device(batch, device)

                logits = attacked_model(batch)
                loss = F.cross_entropy(logits, batch.y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if enable_prints:
                print(
                    f"[Blind FT] Epoch {epoch + 1:02d}/{epochs} | "
                    f"Loss: {total_loss / len(loader):.4f}"
                )

        return attacked_model

    def blind_pruning_attack(self, model, pruning_rate=0.2):
        attacked_model = copy.deepcopy(model)

        for _, module in attacked_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=pruning_rate)
                prune.remove(module, "weight")

        return attacked_model

    def informed_fine_tune_attack(
        self,
        model,
        clean_dataset,
        watermark_graphs,
        enable_prints=False,
        epochs=10,
        learning_rate=1e-4,
        lambda_adv=0.5,
    ):
        attacked_model = copy.deepcopy(model)
        device = self._get_model_device(attacked_model)

        clean_loader = DataLoader(clean_dataset, batch_size=self.batch_size, shuffle=True)
        watermark_loader = DataLoader(watermark_graphs, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(attacked_model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            attacked_model.train()
            total_loss = 0.0
            clean_loss_total = 0.0
            adv_loss_total = 0.0

            watermark_iter = iter(watermark_loader)

            for clean_batch in clean_loader:
                clean_batch = self._move_batch_to_device(clean_batch, device)

                try:
                    watermark_batch = next(watermark_iter)
                except StopIteration:
                    watermark_iter = iter(watermark_loader)
                    watermark_batch = next(watermark_iter)

                watermark_batch = self._move_batch_to_device(watermark_batch, device)

                clean_logits = attacked_model(clean_batch)
                clean_loss = F.cross_entropy(clean_logits, clean_batch.y)

                watermark_logits = attacked_model(watermark_batch)
                watermark_probs = torch.softmax(watermark_logits, dim=1)

                # Reduce the model's confidence on known watermark graphs
                adv_loss = watermark_probs.max(dim=1).values.mean()

                loss = clean_loss + lambda_adv * adv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                clean_loss_total += clean_loss.item()
                adv_loss_total += adv_loss.item()

            if enable_prints:
                print(
                    f"[Informed FT] Epoch {epoch + 1:02d}/{epochs} | "
                    f"Total: {total_loss / len(clean_loader):.4f} | "
                    f"Clean: {clean_loss_total / len(clean_loader):.4f} | "
                    f"Adv: {adv_loss_total / len(clean_loader):.4f}"
                )

        return attacked_model

    def _collect_linear_importance(
        self,
        model,
        dataset,
        objective,
        max_batches=5,
    ):
        device = self._get_model_device(model)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        linear_modules = []
        importance = {}

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules.append((name, module))
                importance[name] = torch.zeros_like(module.weight, device=device)

        model.train()

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            batch = self._move_batch_to_device(batch, device)
            model.zero_grad()

            logits = model(batch)

            if objective == "watermark":
                probs = torch.softmax(logits, dim=1)
                loss = probs.max(dim=1).values.mean()
            elif objective == "clean":
                loss = F.cross_entropy(logits, batch.y)
            else:
                raise ValueError("objective must be 'watermark' or 'clean'")

            loss.backward()

            for name, module in linear_modules:
                if module.weight.grad is not None:
                    importance[name] += module.weight.grad.detach().abs()

        model.zero_grad()
        return importance

    def informed_pruning_attack(
        self,
        model,
        clean_dataset,
        watermark_graphs,
        pruning_rate=0.2,
        clean_preservation_weight=0.5,
        max_importance_batches=5,
    ):
        attacked_model = copy.deepcopy(model)

        watermark_importance = self._collect_linear_importance(
            model=attacked_model,
            dataset=watermark_graphs,
            objective="watermark",
            max_batches=max_importance_batches,
        )

        clean_importance = self._collect_linear_importance(
            model=attacked_model,
            dataset=clean_dataset,
            objective="clean",
            max_batches=max_importance_batches,
        )

        for name, module in attacked_model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            wm_imp = watermark_importance[name]
            clean_imp = clean_importance[name]

            # prune weights that matter for watermark behaviour,
            # but try to preserve weights important for clean classification
            attack_score = wm_imp - clean_preservation_weight * clean_imp

            flat_score = attack_score.flatten()
            num_weights = flat_score.numel()
            num_to_prune = int(pruning_rate * num_weights)

            if num_to_prune <= 0:
                continue

            prune_indices = torch.topk(flat_score, k=num_to_prune).indices

            flat_mask = torch.ones_like(flat_score, device=flat_score.device)
            flat_mask[prune_indices] = 0.0
            mask = flat_mask.view_as(module.weight)

            prune.custom_from_mask(module, name="weight", mask=mask)
            prune.remove(module, "weight")

        return attacked_model