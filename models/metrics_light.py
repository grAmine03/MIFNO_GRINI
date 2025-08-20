import os
import glob
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import json
import timeit
import time
import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ffno_model import FFNO_3D
from mifno_model import MIFNO_3D
from maskfno_model import maskMIFNO_3D
from data_loader import GeologyTracesSourceMaskDataset
from utils_metrics import fourier_spectra

torch.set_float32_matmul_precision("high")


# ---- metrics helpers ----
def myL1(x, axis=0):
    return np.mean(np.abs(x), axis=axis)


def myL2(x, axis=0):
    return np.sqrt(np.mean(x**2, axis=axis))


# ---- ckpt loading helpers ----
def _strip_prefix(k: str) -> str:
    for p in ("model.", "module.", "net.", "model.model."):
        if k.startswith(p):
            return k[len(p):]
    return k


def load_weights(model: torch.nn.Module, path: str, device: torch.device):
    """
    Robust loader for PyTorch 2.6+:
    1) Try safe load with weights_only=True and allowlist argparse.Namespace.
    2) Fallback to weights_only=False (trusted ckpt) if step 1 fails.
    """
    state = None
    # Step 1: safe, weights-only
    try:
        # allowlist needed by Lightning ckpts (they often store argparse.Namespace in hyper_parameters)
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Safe weights-only load failed: {e}")
        print("Falling back to weights_only=False (only do this for trusted checkpoints).")
        # Step 2: full unpickle (trusted)
        state = torch.load(path, map_location=device, weights_only=False)

    # If it’s a Lightning ckpt, extract the actual weights
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    tgt = model.state_dict()
    new_state = {}
    for k, v in state.items():
        k2 = _strip_prefix(k)
        if k2 in tgt and isinstance(v, torch.Tensor):
            new_state[k2] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"Loaded {len(new_state)} tensors from {os.path.basename(path)}")
    if missing:
        print(f"Missing keys (truncated): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys (truncated): {unexpected[:10]}")

def _safe_write_h5(h5_path: str, arrays: dict):
        """
        Try writing HDF5 once; on lock error, write to a unique filename.
        """
        try:
            with h5py.File(h5_path, "w") as f:
                for k, v in arrays.items():
                    f.create_dataset(k, data=v)
            return h5_path
        except (BlockingIOError, OSError) as e:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            root, ext = os.path.splitext(h5_path)
            alt_path = f"{root}-{stamp}{ext}"
            with h5py.File(alt_path, "w") as f:
                for k, v in arrays.items():
                    f.create_dataset(k, data=v)
            return alt_path
# ---- Lightning Eval Module ----
class EvalModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, dt: float, save_cfg: dict):
        super().__init__()
        self.model = model
        self.dt = dt
        self.save_cfg = save_cfg
        self.automatic_optimization = False
        # buffers
        self._uE, self._uN, self._uZ = [], [], []
        self._outE, self._outN, self._outZ = [], [], []
        self._norm = []

    def forward(self, a, s, grid_bounds=None):
        try:
            return self.model(a, s, grid_bounds)
        except TypeError:
            return self.model(a, s)

    def test_step(self, batch, batch_idx):
        # Expected dataset tuple: (a, uE, uN, uZ, s, grid_bounds, norm_cst)
        a, uE, uN, uZ, s, grid_bounds, norm_cst = batch
        with torch.no_grad():
            outE, outN, outZ = self(a, s, grid_bounds)

        # Strip last channel dim
        self._uE.append(uE[..., 0].detach().cpu())
        self._uN.append(uN[..., 0].detach().cpu())
        self._uZ.append(uZ[..., 0].detach().cpu())
        self._outE.append(outE[..., 0].detach().cpu())
        self._outN.append(outN[..., 0].detach().cpu())
        self._outZ.append(outZ[..., 0].detach().cpu())
        self._norm.append(norm_cst.view(-1, 1, 1, 1).detach().cpu())

    def on_test_start(self):
        self._t0 = timeit.default_timer()

    
    
    def on_test_end(self):
        if getattr(self, "global_rank", 0) != 0:
            return

        # Disable HDF5 file locking on Lustre/NFS
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

        elapsed = timeit.default_timer() - self._t0
        print(f"time to compute outputs: {elapsed:.4f}s")

        # Stack to numpy
        all_uE = torch.cat(self._uE, 0).numpy()
        all_uN = torch.cat(self._uN, 0).numpy()
        all_uZ = torch.cat(self._uZ, 0).numpy()
        all_outE = torch.cat(self._outE, 0).numpy()
        all_outN = torch.cat(self._outN, 0).numpy()
        all_outZ = torch.cat(self._outZ, 0).numpy()
        all_norm = torch.cat(self._norm, 0).numpy()

        Ntest = all_uE.shape[0]

        # rMAE / rRMSE on normalized quantities (as in original)
        eps = 1e-2
        base_E = np.abs(all_uE) + eps
        base_N = np.abs(all_uN) + eps
        base_Z = np.abs(all_uZ) + eps

        rMAE = myL1(all_outE / base_E - all_uE / base_E, axis=-1)
        rMAE += myL1(all_outN / base_N - all_uN / base_N, axis=-1)
        rMAE += myL1(all_outZ / base_Z - all_uZ / base_Z, axis=-1)
        rMAE /= 3.0

        rRMSE = myL2(all_outE / base_E - all_uE / base_E, axis=-1)
        rRMSE += myL2(all_outN / base_N - all_uN / base_N, axis=-1)
        rRMSE += myL2(all_outZ / base_Z - all_uZ / base_Z, axis=-1)
        rRMSE /= 3.0

        # Undo trace normalization
        all_uE = all_uE / all_norm
        all_uN = all_uN / all_norm
        all_uZ = all_uZ / all_norm
        all_outE = all_outE / all_norm
        all_outN = all_outN / all_norm
        all_outZ = all_outZ / all_norm

        # Frequency biases
        dt = self.dt
        low_uE, mid_uE, high_uE = fourier_spectra(all_uE, (0, 1), (1, 2), (2, 5), dt=dt)
        low_uN, mid_uN, high_uN = fourier_spectra(all_uN, (0, 1), (1, 2), (2, 5), dt=dt)
        low_uZ, mid_uZ, high_uZ = fourier_spectra(all_uZ, (0, 1), (1, 2), (2, 5), dt=dt)
        low_u = (low_uE + low_uN + low_uZ) / 3.0
        mid_u = (mid_uE + mid_uN + mid_uZ) / 3.0
        high_u = (high_uE + high_uN + high_uZ) / 3.0

        low_oE, mid_oE, high_oE = fourier_spectra(all_outE, (0, 1), (1, 2), (2, 5), dt=dt)
        low_oN, mid_oN, high_oN = fourier_spectra(all_outN, (0, 1), (1, 2), (2, 5), dt=dt)
        low_oZ, mid_oZ, high_oZ = fourier_spectra(all_outZ, (0, 1), (1, 2), (2, 5), dt=dt)
        low_o = (low_oE + low_oN + low_oZ) / 3.0
        mid_o = (mid_oE + mid_oN + mid_oZ) / 3.0
        high_o = (high_oE + high_oN + high_oZ) / 3.0

        fourier_bias_low = (low_o - low_u) / (low_u + 1e-12)
        fourier_bias_mid = (mid_o - mid_u) / (mid_u + 1e-12)
        fourier_bias_high = (high_o - high_u) / (high_u + 1e-12)

        # Assemble DataFrame
        df = pd.DataFrame(index=np.arange(Ntest),
                          columns=["rMAE", "rRMSE", "rFFTlow", "rFFTmid", "rFFThigh", "EG", "PG"],
                          dtype=float)
        df.loc[:, "rMAE"] = np.mean(rMAE, axis=(1, 2))
        df.loc[:, "rRMSE"] = np.mean(rRMSE, axis=(1, 2))
        df.loc[:, "rFFTlow"] = np.mean(fourier_bias_low, axis=(1, 2))
        df.loc[:, "rFFTmid"] = np.mean(fourier_bias_mid, axis=(1, 2))
        df.loc[:, "rFFThigh"] = np.mean(fourier_bias_high, axis=(1, 2))

        summary = pd.DataFrame(index=["mean", "std", "q1", "q3"], columns=df.columns, dtype=float)
        summary.loc["mean"] = df.mean(axis=0)
        summary.loc["std"] = df.std(axis=0)
        summary.loc["q1"] = df.quantile(0.25, axis=0)
        summary.loc["q3"] = df.quantile(0.75, axis=0)

        df = pd.concat([summary, df], axis=0)

        # Save
        dir_logs = self.save_cfg["dir_logs"]
        name_config = self.save_cfg["name_config"]
        epochs = self.save_cfg["epochs"]
        data_type = self.save_cfg["data_type"]

        os.makedirs(os.path.join(dir_logs, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(dir_logs, "outputs"), exist_ok=True)

        csv_path = os.path.join(dir_logs, "metrics", f"metrics_{name_config}-epochs{epochs}{data_type}.csv")
        df.to_csv(csv_path)
        print(summary)

        base_h5 = f"outputs-{name_config}-epochs{epochs}{data_type}.h5"
        h5_path = os.path.join(dir_logs, "outputs", base_h5)
        if os.path.exists(h5_path):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            h5_path = os.path.join(dir_logs, "outputs", f"{base_h5[:-3]}-{stamp}.h5")

        h5_path = _safe_write_h5(
            h5_path,
            {
                "uE": all_uE,
                "uN": all_uN,
                "uZ": all_uZ,
                "outE": all_outE,
                "outN": all_outN,
                "outZ": all_outZ,
            },
        )

        print(f"Saved metrics to: {csv_path}")
        print(f"Saved outputs to: {h5_path}")

    def configure_optimizers(self):
        return None

    
# ---- CLI: reuse train_lightning_fixed args, plus test/ckpt/dt ----
def parse_args():
    parser = argparse.ArgumentParser()
    # same as train_lightning_fixed.py
    parser.add_argument('--model_type', type=str, default="maskMIFNO")
    parser.add_argument('--S_in', type=int, default=32)
    parser.add_argument('--S_in_z', type=int, default=32)
    parser.add_argument('--S_out', type=int, default=32)
    parser.add_argument('--T_out', type=int, default=320)
    parser.add_argument('--nlayers', type=int, default=16)
    parser.add_argument('--branching_index', type=int, default=4)
    parser.add_argument('--dv', type=int, default=16)
    parser.add_argument('--list_dv', type=int, nargs='+')
    parser.add_argument('--list_D1', type=int, nargs='+', default=[32]*16)
    parser.add_argument('--list_D2', type=int, nargs='+', default=[32]*16)
    parser.add_argument('--list_D3', type=int, nargs='+', default=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 128, 256, 320])
    parser.add_argument('--list_M1', type=int, nargs='+', default=[16]*16)
    parser.add_argument('--list_M2', type=int, nargs='+', default=[16]*16)
    parser.add_argument('--list_M3', type=int, nargs='+', default=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32])
    parser.add_argument('--Ntrain', type=int, default=27000)
    parser.add_argument('--Nval', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--source_orientation', type=str, default='angle')
    parser.add_argument('--normalize_source', action='store_true')
    parser.add_argument('--normalize_traces', action='store_true')
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--learning_rate', type=float, default=0.0006)
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.0, 0.0])
    parser.add_argument('--dir_data_train', type=str, nargs='+', default=['HEMEWS3D_S32_Z32_T320_fmax5_rot0_train'])
    parser.add_argument('--dir_data_val', type=str, nargs='+', default=['HEMEWS3D_S32_Z32_T320_fmax5_rot0_val'])
    parser.add_argument('--dir_logs', type=str, default='/lustre/fsn1/projects/rech/xvy/upz57sx/MIFNO_logs/')
    parser.add_argument('--additional_name', type=str, default="")
    parser.add_argument('--restart_model', action='store_true', default=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_plot_every_n_epochs', type=int, default=10)

    # evaluation-specific additions
    parser.add_argument('--dir_data_test', type=str, nargs='+', default=['HEMEWS3D_S32_Z32_T320_fmax5_rot0_val'])
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to .ckpt or .pt')
    parser.add_argument('--dt', type=float, default=0.02, help='Sampling period for FFT metrics')

    return parser.parse_args()


def build_model_from_options(opt):
    if opt.model_type == 'MIFNO':
        input_dim = 4
        output_dim = 1
        source_dim = 6 if opt.source_orientation == 'angle' else 9
        return MIFNO_3D(
            np.array(opt.list_D1), np.array(opt.list_D2), np.array(opt.list_D3),
            np.array(opt.list_M1), np.array(opt.list_M2), np.array(opt.list_M3),
            opt.dv,
            input_dim=input_dim, output_dim=output_dim, source_dim=source_dim,
            n_layers=opt.nlayers, branching_index=opt.branching_index, padding=opt.padding
        )
    elif opt.model_type == 'FFNO':
        input_dim = 10 if opt.source_orientation == 'angle' else 13
        output_dim = 1
        list_dv = None if opt.list_dv is None else np.array(opt.list_dv).astype(int)
        return FFNO_3D(
            np.array(opt.list_D1), np.array(opt.list_D2), np.array(opt.list_D3),
            np.array(opt.list_M1), np.array(opt.list_M2), np.array(opt.list_M3),
            opt.dv,
            input_dim=input_dim, output_dim=output_dim,
            list_width=list_dv if list_dv is not None else np.array([opt.dv]*opt.nlayers),
            n_layers=opt.nlayers, padding=opt.padding
        )
    elif opt.model_type == 'maskMIFNO':
        input_dim = 4
        output_dim = 1
        source_dim = 6 if opt.source_orientation == 'angle' else 9
        
        return maskMIFNO_3D(
            np.array(opt.list_D1), np.array(opt.list_D2), np.array(opt.list_D3),
            np.array(opt.list_M1), np.array(opt.list_M2), np.array(opt.list_M3),
            opt.dv,
            input_dim=input_dim, output_dim=output_dim, source_dim=source_dim,
            n_layers=opt.nlayers, branching_index=opt.branching_index, padding=opt.padding, time_emb_dim=16
        )
        
    else:
        raise ValueError(f"Unknown model_type: {opt.model_type}")


if __name__ == "__main__":
    opt = parse_args()

    # Build name_config exactly like training
    name_config = (
        "JeanZay_FixedBounds-"
        f"{opt.model_type}3D-{opt.source_orientation}-"
        f"dv{opt.dv}-{opt.nlayers}layers-S{opt.S_in}-T{opt.T_out}-"
        f"learningrate{str(opt.learning_rate).replace('.','p')}-Ntrain{opt.Ntrain}-"
        f"batchsize{opt.batch_size}-"
    )
    if opt.normalize_source:
        name_config += "-normedsource"
    if opt.normalize_traces:
        name_config += "-normedtraces"
    name_config += opt.additional_name

    # Dataset (test)
    test_ds = GeologyTracesSourceMaskDataset(
        path_data='/lustre/fsn1/projects/rech/xvy/upz57sx/hemews3d/formatted/',
        dir_data=opt.dir_data_test,
        S_in=opt.S_in,
        S_in_z=opt.S_in_z,
        S_out=opt.S_out,
        T_out=opt.T_out,
        transform_a='normal',
        N=None,  # use full test set
        orientation=opt.source_orientation,
        transform_position=[9600, 9600, 9600],
        transform_angle='unit',
        transform_traces=None
    )
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, shuffle=False, num_workers=2)

    # Model
    model = build_model_from_options(opt)

    # Resolve checkpoint
    ckpt_path = opt.ckpt_path
    if ckpt_path is None:
        # Prefer best-validation Lightning ckpt saved during training
        #pattern = os.path.join(opt.dir_logs, f"model_best_val_ckpt-{name_config}-*.ckpt")
        pattern = os.path.join(opt.dir_logs, f"model_best_val_ckpt-Tbranch_FixedBounds-maskMIFNO3D-angle-dv16-16layers-S32-T320-learningrate0p0004-Ntrain27000-batchsize16--normedtraces-epoch=323-val_loss=0.70.ckpt")
        print("Checkpoint pattern:", pattern )
        cands = sorted(glob.glob(pattern))
        if cands:
            ckpt_path = cands[-1]
        else:
            # Fallback to legacy .pt
            pt_try = os.path.join(opt.dir_logs, f"models/bestmodel-{name_config}-epochs{opt.epochs}.pt")
            if os.path.exists(pt_try):
                ckpt_path = pt_try

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        load_weights(model, ckpt_path, device=torch.device("cpu"))
    else:
        print("Warning: no checkpoint found, running with randomly initialized weights.")

    # data_type suffix
    first = (opt.dir_data_test[0] if len(opt.dir_data_test) else "").lower()
    if first.endswith("train"):
        data_type = "_train"
    elif first.endswith("val"):
        data_type = "_val"
    elif first.endswith("test"):
        data_type = "_test"
    else:
        data_type = ""

    lit = EvalModule(model=model, dt=opt.dt,
                     save_cfg=dict(dir_logs=opt.dir_logs, name_config=name_config,
                                   epochs=opt.epochs, data_type=data_type))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False, enable_checkpointing=False)
    trainer.test(lit, dataloaders=test_loader)