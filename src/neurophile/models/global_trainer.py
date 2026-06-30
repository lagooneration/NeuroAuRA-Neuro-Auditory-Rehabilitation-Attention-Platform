"""
neurophile.models.global_trainer
================================
Strategy-Pattern training orchestrator for the Global CI Foundation Model.

Strategy Pattern
----------------
``GlobalCITrainer`` accepts any model that implements *either* the PyTorch
``BaseAADModel`` (``nn.Module``) interface *or* the scikit-learn ``BaseDecoder``
(``fit/predict``) interface. On instantiation, it inspects the model type and
selects the appropriate training strategy:

  - ``_PyTorchStrategy``  : batched backward pass, gradient checkpointing,
                            yields checkpoints compatible with Flower aggregation.
  - ``_SklearnStrategy``  : direct ``fit()`` call, no epochs, CPU-bound.

The training loop follows the six-step mathematical progression from the
blueprint:
  1. Ingest baseline audio
  2. Vocode → CI acoustic simulation
  3. Extract low-frequency envelope (0.5–8 Hz)
  4. Clean EEG (CI artifact cancellation)
  5. Feed (clean EEG, CI envelope) into the adapter model
  6. Compute Pearson correlation loss; backpropagate (PyTorch) or fit (sklearn)

Loss Function
-------------
For deep learning models, the loss is the *negative Pearson correlation*
between the model's predicted envelope and the true CI envelope. This is
mathematically equivalent to maximising cortical tracking, which is the
biological target in AAD.

    L = 1 - ρ(ŷ, y)

where ŷ = sigmoid(model(EEG, envelope)) and y = binary attention label, OR
for reconstruction-style training: ŷ = model output, y = envelope amplitude.

This implementation supports **classification mode** (binary CE loss) and
**reconstruction mode** (Pearson correlation loss) via the ``loss_mode`` arg.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Internal strategies ───────────────────────────────────────────────────────

class _PyTorchStrategy:
    """Training strategy for ``nn.Module`` (``BaseAADModel``) models."""

    def __init__(
        self,
        model: "nn.Module",
        lr: float,
        loss_mode: str,
        device: str,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.loss_mode = loss_mode
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        if loss_mode == "classification":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = self._pearson_loss

    @staticmethod
    def _pearson_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        """Differentiable negative Pearson correlation loss."""
        pred_c = pred - pred.mean()
        tgt_c = target - target.mean()
        num = (pred_c * tgt_c).sum()
        denom = torch.sqrt((pred_c ** 2).sum() * (tgt_c ** 2).sum()) + 1e-8
        return 1.0 - num / denom

    def fit(
        self,
        eeg_array: np.ndarray,
        envelope_array: np.ndarray,
        label_array: np.ndarray,
        epochs: int,
        batch_size: int,
        val_eeg: np.ndarray | None = None,
        val_env: np.ndarray | None = None,
        val_lbl: np.ndarray | None = None,
    ) -> list[float]:
        """Run the PyTorch training loop with Early Stopping."""
        import copy

        eeg_t = torch.from_numpy(eeg_array).float()
        env_t = torch.from_numpy(envelope_array).float()
        lbl_t = torch.from_numpy(label_array).float().unsqueeze(1)

        dataset = TensorDataset(eeg_t, env_t, lbl_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        has_val = val_eeg is not None and val_env is not None and val_lbl is not None
        if has_val:
            v_eeg_t = torch.from_numpy(val_eeg).float().to(self.device)
            v_env_t = torch.from_numpy(val_env).float().to(self.device)
            v_lbl_t = torch.from_numpy(val_lbl).float().unsqueeze(1).to(self.device)

        history: list[float] = []
        best_val_loss = float("inf")
        best_weights = None
        patience = 5
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()
            for eeg_b, env_b, lbl_b in loader:
                eeg_b = eeg_b.to(self.device)
                env_b = env_b.to(self.device)
                lbl_b = lbl_b.to(self.device)

                self.optimizer.zero_grad()
                logit = self.model(eeg_b, env_b)

                if self.loss_mode == "classification":
                    loss = self.criterion(logit, lbl_b)
                else:
                    loss = self._pearson_loss(
                        torch.sigmoid(logit), lbl_b
                    )

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            mean_loss = epoch_loss / max(len(loader), 1)
            history.append(mean_loss)
            elapsed = time.time() - t0

            # Early Stopping Check
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_logit = self.model(v_eeg_t, v_env_t)
                    if self.loss_mode == "classification":
                        val_loss = self.criterion(v_logit, v_lbl_t).item()
                    else:
                        val_loss = self._pearson_loss(torch.sigmoid(v_logit), v_lbl_t).item()

                logger.info(
                    "Epoch %3d/%d | train_loss=%.5f | val_loss=%.5f | %.1fs",
                    epoch, epochs, mean_loss, val_loss, elapsed,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info("Early stopping triggered! Restoring best weights (Epoch %d)", epoch - patience)
                        self.model.load_state_dict(best_weights)
                        break
            else:
                logger.info(
                    "Epoch %3d/%d | loss=%.5f | %.1fs",
                    epoch, epochs, mean_loss, elapsed,
                )

        return history

    def save_checkpoint(self, path: Path, meta: dict) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "meta": meta,
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)


class _SklearnStrategy:
    """Training strategy for ``BaseDecoder`` (scikit-learn) models."""

    def __init__(self, model: object) -> None:
        self.model = model

    def fit(
        self,
        eeg_array: np.ndarray,
        envelope_array: np.ndarray,
        label_array: np.ndarray,  # noqa: ARG002
        epochs: int,              # noqa: ARG002
        batch_size: int,          # noqa: ARG002
    ) -> list[float]:
        """Call model.fit() — sklearn models train in a single call.

        The mTRF backward model (LinearDecoder) requires minutes of continuous
        EEG data to accumulate meaningful covariance matrices. We concatenate
        all attended-stream trials (label==1) into a single continuous segment
        before calling fit(). This is the standard mTRF training protocol.
        """
        # Select only attended-stream trials (label == 1.0) for fitting
        # the backward model: EEG -> attended envelope
        attended_mask = label_array == 1.0
        if attended_mask.sum() == 0:
            attended_mask = np.ones(len(label_array), dtype=bool)  # fallback

        eeg_2d = np.concatenate(eeg_array[attended_mask], axis=0)        # (T_total, C)
        env_1d = np.concatenate(envelope_array[attended_mask, :, 0], axis=0)  # (T_total,)

        logger.info(
            "SklearnStrategy: training on %d attended trials → %d total samples",
            attended_mask.sum(), len(eeg_2d),
        )

        # Infer fs from caller context (stored on model after first fit, else default)
        fs = getattr(self.model, "fs", 64)
        if fs == 0:
            fs = 64
        self.model.fit(eeg_2d, env_1d, fs=fs)
        score = self.model.score(eeg_2d, env_1d)
        logger.info("sklearn model fitted. Score (ρ) = %.4f", score)
        return [1.0 - score]  # pseudo-loss for consistency

    def save_checkpoint(self, path: Path, meta: dict) -> None:
        import pickle
        payload = {"model": self.model, "meta": meta}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("sklearn checkpoint saved → %s", path)


# ── GlobalCITrainer ───────────────────────────────────────────────────────────

class GlobalCITrainer:
    """Strategy-pattern trainer for the Global CI Foundation Model.

    Accepts any model implementing ``BaseAADModel`` (PyTorch) *or*
    ``BaseDecoder`` (scikit-learn). Routes to the appropriate strategy
    automatically.

    Parameters
    ----------
    model : BaseAADModel or BaseDecoder
        The adapter model to train.
    lr : float
        Learning rate (PyTorch only).
    epochs : int
        Number of training epochs (PyTorch only; sklearn trains in one shot).
    batch_size : int
        Mini-batch size (PyTorch only).
    loss_mode : str
        ``"classification"`` (BCE loss) or ``"reconstruction"`` (Pearson loss).
    device : str
        Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    output_dir : str or Path
        Directory for checkpoints and loss curves.

    Examples
    --------
    >>> from neurophile.models.adapters import KULAdapter
    >>> from neurophile.models.global_trainer import GlobalCITrainer
    >>> import numpy as np
    >>> model = KULAdapter(num_eeg_channels=64)
    >>> trainer = GlobalCITrainer(model, epochs=10)
    >>> # Synthetic data
    >>> eeg = np.random.randn(32, 512, 64).astype("float32")  # (N, T, C)
    >>> env = np.random.randn(32, 512, 1).astype("float32")   # (N, T, 1)
    >>> labels = np.random.randint(0, 2, 32).astype("float32")
    >>> history = trainer.train(eeg, env, labels)
    """

    def __init__(
        self,
        model: object,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        loss_mode: str = "classification",
        device: str = "cpu",
        output_dir: str | Path = "./checkpoints",
    ) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Strategy selection ────────────────────────────────────────────────
        if _TORCH_AVAILABLE and isinstance(model, nn.Module):
            logger.info(
                "GlobalCITrainer: detected PyTorch model (%s) → PyTorchStrategy",
                type(model).__name__,
            )
            self._strategy: _PyTorchStrategy | _SklearnStrategy = _PyTorchStrategy(
                model=model,
                lr=lr,
                loss_mode=loss_mode,
                device=device,
            )
            self._backend = "pytorch"
        else:
            # Fallback to sklearn strategy (BaseDecoder)
            logger.info(
                "GlobalCITrainer: detected sklearn model (%s) → SklearnStrategy",
                type(model).__name__,
            )
            self._strategy = _SklearnStrategy(model=model)
            self._backend = "sklearn"

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        eeg_array: np.ndarray,
        envelope_array: np.ndarray,
        label_array: np.ndarray,
        val_eeg: np.ndarray | None = None,
        val_env: np.ndarray | None = None,
        val_lbl: np.ndarray | None = None,
    ) -> list[float]:
        """Run training on pre-cleaned EEG + CI envelope.

        This method is the Step 5+6 of the blueprint's 6-step pipeline:
        the upstream script (``train_global_ci_model.py``) handles Steps 1–4.

        Parameters
        ----------
        eeg_array : np.ndarray, shape (N, T, C)
            Cleaned EEG trials. N=trials, T=time, C=channels.
        envelope_array : np.ndarray, shape (N, T, 1)
            Simulated CI envelope (output of ``CIVocoderSimulator`` + envelope).
        label_array : np.ndarray, shape (N,)
            Binary labels: 1 = attended stream, 0 = unattended.
        val_eeg, val_env, val_lbl : np.ndarray | None
            Optional validation data for early stopping (PyTorch only).

        Returns
        -------
        loss_history : list[float]
            Per-epoch mean loss (single value for sklearn).
        """
        logger.info(
            "GlobalCITrainer.train(): N=%d trials, T=%d steps, C=%d channels | backend=%s",
            eeg_array.shape[0], eeg_array.shape[1], eeg_array.shape[2], self._backend,
        )

        if self._backend == "pytorch":
            history = self._strategy.fit(
                eeg_array=eeg_array,
                envelope_array=envelope_array,
                label_array=label_array,
                epochs=self.epochs,
                batch_size=self.batch_size,
                val_eeg=val_eeg,
                val_env=val_env,
                val_lbl=val_lbl,
            )
        else:
            history = self._strategy.fit(
                eeg_array=eeg_array,
                envelope_array=envelope_array,
                label_array=label_array,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )

        # Auto-save final checkpoint
        model_name = getattr(self.model, "name", "model")
        ckpt_path = self.output_dir / f"{model_name}_global_ci.pt"
        self._strategy.save_checkpoint(
            ckpt_path,
            meta={
                "model_name": model_name,
                "backend": self._backend,
                "n_trials": len(eeg_array),
                "final_loss": history[-1] if history else None,
            },
        )
        return history

    def evaluate(
        self,
        eeg_array: np.ndarray,
        envelope_array: np.ndarray,
        label_array: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model accuracy on held-out data.

        Returns
        -------
        metrics : dict with keys ``accuracy``, ``mean_pearson_r``
        """
        if self._backend == "pytorch":
            return self._eval_pytorch(eeg_array, envelope_array, label_array)
        return self._eval_sklearn(eeg_array, envelope_array, label_array)

    def _eval_pytorch(
        self,
        eeg_array: np.ndarray,
        envelope_array: np.ndarray,
        label_array: np.ndarray,
    ) -> dict[str, float]:
        self._strategy.model.eval()  # type: ignore[attr-defined]
        device = self._strategy.device  # type: ignore[attr-defined]
        all_probs: list[float] = []
        all_labels: list[float] = []

        from torch.utils.data import DataLoader, TensorDataset
        eeg_t = torch.from_numpy(eeg_array).float()
        env_t = torch.from_numpy(envelope_array).float()
        lbl_t = torch.from_numpy(label_array).float()
        dataset = TensorDataset(eeg_t, env_t, lbl_t)
        # Use the configured batch_size, or a large default for eval
        eval_batch_size = getattr(self, "batch_size", 32)
        loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

        with torch.no_grad():
            for eeg_b, env_b, lbl_b in loader:
                eeg_b = eeg_b.to(device)
                env_b = env_b.to(device)
                logit = self._strategy.model(eeg_b, env_b)  # type: ignore[attr-defined]
                prob = torch.sigmoid(logit).squeeze(-1)
                all_probs.extend(prob.cpu().tolist())
                all_labels.extend(lbl_b.tolist())

        # 2AFC Evaluation: compare probability of attended vs unattended for each pair
        correct_2afc = 0
        total_2afc = 0
        
        for i in range(0, len(all_probs) - 1, 2):
            prob_att = all_probs[i]
            prob_unatt = all_probs[i + 1]
            if prob_att > prob_unatt:
                correct_2afc += 1
            total_2afc += 1
            
        accuracy = correct_2afc / max(total_2afc, 1)

        # Compute Pearson r over all predictions at once (needs ≥2 unique values)
        mean_pearson_r = 0.0
        if len(all_probs) >= 2:
            probs_arr = np.array(all_probs)
            labels_arr = np.array(all_labels)
            # Only compute if there is variance in both arrays
            if probs_arr.std() > 1e-9 and labels_arr.std() > 1e-9:
                r = np.corrcoef(probs_arr, labels_arr)[0, 1]
                if np.isfinite(r):
                    mean_pearson_r = float(r)

        return {
            "accuracy": accuracy,
            "mean_pearson_r": mean_pearson_r,
        }

    def _eval_sklearn(
        self,
        eeg_array: np.ndarray,
        envelope_array: np.ndarray,
        label_array: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluate using 2AFC: for each attended/unattended trial pair at the
        same time index, compare reconstruction correlations to both envelopes.
        Accuracy = fraction of trials where r(attended) > r(unattended).
        This matches the evaluation protocol used in the AAD literature.
        """
        n = len(eeg_array)
        # Pair up attended (label=1) and unattended (label=0) trials
        # Trials are generated in pairs: [att, unatt, att, unatt, ...]
        correct = 0
        total = 0
        all_r_att = []

        for i in range(0, n - 1, 2):
            eeg_2d = eeg_array[i]             # same EEG for both in the pair
            env_att = envelope_array[i, :, 0]
            env_unatt = envelope_array[i + 1, :, 0]

            r_att = self.model.score(eeg_2d, env_att)
            r_unatt = self.model.score(eeg_2d, env_unatt)

            if r_att > r_unatt:
                correct += 1
            total += 1
            all_r_att.append(r_att)

        mean_r = float(np.mean(all_r_att)) if all_r_att else 0.0
        accuracy = correct / max(total, 1)
        return {
            "accuracy": accuracy,
            "mean_pearson_r": mean_r,
        }
