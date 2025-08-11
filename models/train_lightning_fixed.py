import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import random
import io # Add io import
from PIL import Image # Add PIL import
import os
from utils_models import get_device, get_batch_size, loss_criterion, RunningAverage
from ffno_model import FFNO_3D
from mifno_model import MIFNO_3D
from maskfno_model import maskMIFNO_3D
from data_loader import GeologyTracesSourceMaskDataset
from dataloaders import GeologyTracesSourceDataset
import idr_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


torch.set_float32_matmul_precision('high') 


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="maskMIFNO", help="Architecture used: MIFNO or F-FNO")
parser.add_argument('--S_in', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('--S_in_z', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('--S_out', type=int, default=32, help="Size of the spatial output grid")
parser.add_argument('--T_out', type=int, default=320, help="Number of time steps")
parser.add_argument('--nlayers', type=int, default=16, help="Number of layers")
parser.add_argument('--branching_index', type=int, default=4, help="Index of the first FFNO block seeing the source")
parser.add_argument('--dv', type=int, default=16, help="Number of channels")
parser.add_argument('--list_dv', type=int, nargs='+', help = "Number of channels in uplfit block + each Fourier block (used only in F-FNO, not in MIFNO)")
parser.add_argument('--list_D1', type=int, nargs='+', default=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], help = "Dimensions along the 1st dimension after each block")
parser.add_argument('--list_D2', type=int, nargs='+', default=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32], help = "Dimensions along the 2nd dimension after each block")
parser.add_argument('--list_D3', type=int, nargs='+', default=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 128, 256, 320], help = "Dimensions along the 3rd dimension after each block")
parser.add_argument('--list_M1', type=int, nargs='+', default=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], help = "Number of modes along the 1st dimension after each block")
parser.add_argument('--list_M2', type=int, nargs='+', default=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], help = "Number of modes along the 2nd dimension after each block")
parser.add_argument('--list_M3', type=int, nargs='+', default=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32], help = "Number of modes along the 3rd dimension after each block")
parser.add_argument('--Ntrain', type=int, default=27000, help="Number of training samples")
parser.add_argument('--Nval', type=int, default=3000, help="Number of validation samples")
parser.add_argument('--batch_size', type=int, default=16, help = 'batch size')
parser.add_argument('--source_orientation', type=str, default='angle', help="angle or moment")
parser.add_argument('--normalize_source', action='store_true', help='Whether to normalize the source position and the angles (if applicable)')
parser.add_argument('--normalize_traces', action='store_true', help='Whether to normalize the traces')
parser.add_argument('--padding', type=int, default=0, help = "Number of pixels for padding on each side of x and y")
parser.add_argument('--epochs', type=int, default=350, help = 'Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.0006, help='learning rate')
parser.add_argument('--loss_weights', type=float, nargs='+', default = [1.0, 0.0], help = "Weight of L1 loss, L2 loss")
parser.add_argument('--dir_data_train', type=str, nargs='+', default=['HEMEWS3D_S32_Z32_T320_fmax5_rot0_train'], help="Name of folders with training data")
parser.add_argument('--dir_data_val', type=str, nargs='+', default=['HEMEWS3D_S32_Z32_T320_fmax5_rot0_val'], help="Name of folders with training data")
parser.add_argument('--dir_logs', type=str, default='/lustre/fsn1/projects/rech/xvy/upz57sx/MIFNO_logs/', help="Path to folder to store loss and models")
parser.add_argument('--additional_name', type=str, default="", help="string to add to the configuration name for saved outputs")
parser.add_argument('--restart_model',action='store_true',default=False,help='Start from checkpoint?')
parser.add_argument('--start_epoch', type=int, default=0, help="Epoch to start, >0 if initializing with a trained model")
parser.add_argument('--seed', type=int, default=0, help="Seed to initialize pytorch")
parser.add_argument('--log_plot_every_n_epochs', type=int, default=10, help="Frequency (in epochs) for logging validation plots") # Add this line
options = parser.parse_args()




# Set seeds and deterministic behavior
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dist.init_process_group(backend='nccl',
                        init_method='env://',
                        world_size=idr_torch.size,
                        rank=idr_torch.rank)
torch.cuda.set_device(idr_torch.local_rank)
gpu = torch.device("cuda")



# LightningModule for the model
class GeologyModel(LightningModule):
    def __init__(self, options):
        super().__init__()
        self.save_hyperparameters()
        self.options = options

        # Model definition remains unchanged
        if options.model_type == 'MIFNO':
            if options.source_orientation == 'angle':
                input_dim = 4  # a, x, y, z
                output_dim = 1
                source_dim = 6  # 3 coordinates + 3 angles
                self.model = MIFNO_3D(
                    np.array(options.list_D1),
                    np.array(options.list_D2), 
                    np.array(options.list_D3),
                    np.array(options.list_M1), 
                    np.array(options.list_M2), 
                    np.array(options.list_M3),
                    options.dv, 
                    input_dim=input_dim, 
                    output_dim=output_dim, 
                    source_dim=source_dim,
                    n_layers=options.nlayers, 
                    branching_index=options.branching_index, 
                    padding=options.padding
                )
            elif options.source_orientation == 'moment':
                input_dim = 4  # a, x, y, z
                output_dim = 1
                source_dim = 9  # 3 coordinates + 6 moment tensor components
                self.model = MIFNO_3D(
                    np.array(options.list_D1), 
                    np.array(options.list_D2), 
                    np.array(options.list_D3),
                    np.array(options.list_M1), 
                    np.array(options.list_M2), 
                    np.array(options.list_M3),
                    options.dv, 
                    input_dim=input_dim, 
                    output_dim=output_dim, 
                    source_dim=source_dim,
                    n_layers=options.nlayers, 
                    branching_index=options.branching_index, 
                    padding=options.padding
                )

        elif options.model_type == 'FFNO':
            list_dv = np.array(options.list_dv).astype(int)
            if options.source_orientation == 'angle':
                input_dim = 10  # a, x, y, z, x_s, y_s, z_s, strike, dip, rake
                output_dim = 1
                self.model = FFNO_3D(
                    np.array(options.list_D1), 
                    np.array(options.list_D2), 
                    np.array(options.list_D3),
                    np.array(options.list_M1), 
                    np.array(options.list_M2), 
                    np.array(options.list_M3),
                    options.dv, 
                    list_dv, 
                    input_dim=input_dim, 
                    output_dim=output_dim,
                    list_width=list_dv, 
                    n_layers=options.nlayers, 
                    padding=options.padding
                )
            elif options.source_orientation == 'moment':
                input_dim = 13  # a, x, y, z, x_s, y_s, z_s, Mxx, Myy, Mzz, Mxy, Mxz, Myz
                output_dim = 1
                self.model = FFNO_3D(
                    np.array(options.list_D1), 
                    np.array(options.list_D2), 
                    np.array(options.list_D3),
                    np.array(options.list_M1), 
                    np.array(options.list_M2), 
                    np.array(options.list_M3),
                    options.dv, 
                    list_dv, 
                    input_dim=input_dim, 
                    output_dim=output_dim,
                    list_width=list_dv, 
                    n_layers=options.nlayers, 
                    padding=options.padding
                )
        elif options.model_type == 'maskMIFNO':
            if options.source_orientation == 'angle':
                input_dim = 4  # a, x, y, z
                output_dim = 1
                source_dim = 6  # 3 coordinates + 3 angles
                self.model = maskMIFNO_3D(
                    np.array(options.list_D1),
                    np.array(options.list_D2), 
                    np.array(options.list_D3),
                    np.array(options.list_M1), 
                    np.array(options.list_M2), 
                    np.array(options.list_M3),
                    options.dv, 
                    input_dim=input_dim, 
                    output_dim=output_dim, 
                    source_dim=source_dim,
                    n_layers=options.nlayers, 
                    branching_index=options.branching_index, 
                    padding=options.padding
                )
            elif options.source_orientation == 'moment':
                input_dim = 4  # a, x, y, z
                output_dim = 1
                source_dim = 9  # 3 coordinates + 6 moment tensor components
                self.model = maskMIFNO_3D(
                    np.array(options.list_D1), 
                    np.array(options.list_D2), 
                    np.array(options.list_D3),
                    np.array(options.list_M1), 
                    np.array(options.list_M2), 
                    np.array(options.list_M3),
                    options.dv, 
                    input_dim=input_dim, 
                    output_dim=output_dim,
                    source_dim=source_dim, 
                    n_layers=options.nlayers,
                    branching_index=options.branching_index, 
                    padding=options.padding
                )
        self.loss_weights = options.loss_weights
        self.loss_criterion = loss_criterion
        

    def forward(self, a, s,grid_bounds):
        return self.model(a, s, grid_bounds)

    def training_step(self, batch, batch_idx):
        a, uE, uN, uZ, s, grid_bounds, norm_cst = batch

        # Forward pass
        outE, outN, outZ = self(a, s, grid_bounds)

        # Compute loss
        loss_rel = self.loss_criterion((outE, outN, outZ), (uE, uN, uZ), self.loss_weights, relative=True)
        self.log('train_loss', loss_rel, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log("grad_norm", total_norm)
        # Log metrics to Wandb
        #wandb.log({"train_loss": loss_rel.item()})
        #wandb.log({"grad_norm": total_norm})

        return loss_rel

    def validation_step(self, batch, batch_idx):
        a, uE, uN, uZ, s, grid_bounds, norm_cst = batch
        
        # Forward pass
        outE, outN, outZ = self(a, s, grid_bounds)
       
        # Compute loss
        loss_rel = self.loss_criterion((outE, outN, outZ), (uE, uN, uZ), self.loss_weights, relative=True)
        self.log('val_loss', loss_rel, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        # Log metrics to Wandb
        #wandb.log({"val_loss": loss_rel.item()})

        if self.trainer.is_global_zero and self.current_epoch % self.hparams.options.log_plot_every_n_epochs == 0 and batch_idx == 0: # Log only for the first batch to avoid too many plots
             self.log_validation_plots(batch, outE, outN, outZ)

    def log_validation_plots(self, batch, outE, outN, outZ):
        """Logs comparison plots for a few random samples to Wandb."""
        a, uE_truth, uN_truth, uZ_truth, s, grid_bounds, norm_cst = batch
        batch_size = a.shape[0]
        num_plots = min(5, batch_size) # Plot up to 5 samples

        # Randomly select indices
        indices = random.sample(range(batch_size), num_plots)

        # Define time and coordinate vectors (use hparams for options)
        dt = 0.02
        time_vec = np.linspace(0, (self.hparams.options.T_out - 1) * dt, self.hparams.options.T_out)
        x_coord_vec = np.linspace(0, 9600, self.hparams.options.S_out) # Assuming 9.6km extent
        y_index = self.hparams.options.S_out // 2 # Plot middle y-slice

        wandb_logs = {}
        for i, idx in enumerate(indices):
            # Select one sample and component (e.g., uE)
            output_sample_E = outE[idx].cpu().numpy()
            truth_sample_E = uE_truth[idx].cpu().numpy()
            print(f"output_sample_E: {output_sample_E}, truth_sample_E: {truth_sample_E}")
            # Extract the 2D slice (x_dim, time_dim)
            output_slice_E = output_sample_E[:, y_index, :]
            truth_slice_E = truth_sample_E[:, y_index, :]
            print(f"output_sample_E: {output_sample_E}, truth_sample_E: {truth_sample_E}")

            # --- Plot comparison (truth and prediction) ---
            fig_comparison = plot_comparison_time_vs_x(
                truth_slice_E, output_slice_E, time_vec, x_coord_vec,
                title_prefix=f"Epoch {self.current_epoch} Sample {idx} (uE) y={y_index}"
            )

            if fig_comparison: # Check if figure creation was successful
                try:
                    buf_comparison = io.BytesIO()
                    fig_comparison.savefig(buf_comparison, format='png', bbox_inches='tight', dpi=200) # Added dpi=200
                    buf_comparison.seek(0)
                    img_comparison = Image.open(buf_comparison)
                    wandb_logs[f"val/epoch_{self.current_epoch}/sample_{idx}/comparison_E"] = wandb.Image(img_comparison)
                    buf_comparison.close()
                except Exception as e:
                    print(f"Error processing comparison plot for sample {idx}: {e}")
                finally:
                     plt.close(fig_comparison) # Ensure figure is closed
            else:
                print(f"Warning: Failed to generate comparison plot for sample {idx}")

        # Log all plots for this batch at once if any were generated
        if wandb_logs:
             try:
                 self.logger.experiment.log(wandb_logs)
                 print(f"Logged {len(wandb_logs)} validation comparison plots for epoch {self.current_epoch}.")
             except Exception as e:
                 print(f"Error logging plots to Wandb: {e}")
        else:
             print(f"No validation plots generated or logged for epoch {self.current_epoch}.")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=options.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

def plot_time_vs_x(data_slice, time_vec, x_coord_vec, title="Time vs X-coordinate"):
    """Displays a 2D plot with time on the horizontal axis and x-coordinate on the vertical axis.

    Args:
        data_slice (np.ndarray): 2D numpy array (x_dim, time_dim) to plot.
        time_vec (np.ndarray): 1D array for the time axis.
        x_coord_vec (np.ndarray): 1D array for the x-coordinate axis.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The generated Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data_slice, aspect='auto', origin='lower', cmap='RdBu',
                   extent=[time_vec[0], time_vec[-1], x_coord_vec[0], x_coord_vec[-1]])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X-coordinate (km)') # Assuming km based on previous context
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('Value')


def plot_comparison_time_vs_x(truth_slice, pred_slice, time_vec, x_coord_vec, title_prefix="Comparison"):
    """Displays ground truth and prediction side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True) # 1 row, 2 columns

    # Determine shared color limits
    max_abs_val = max(np.abs(truth_slice).max(), np.abs(pred_slice).max())
    vmin = -max_abs_val
    vmax = max_abs_val
    
    # Plot Ground Truth
    im_truth = axes[0].imshow(truth_slice, aspect='auto', origin='lower', cmap='RdBu',
                              extent=[time_vec[0], time_vec[-1], x_coord_vec[0], x_coord_vec[-1]],
                              vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('X-coordinate (m)')
    axes[0].set_title(f"{title_prefix} - Ground Truth")

    # Plot Prediction
    im_pred = axes[1].imshow(pred_slice, aspect='auto', origin='lower', cmap='RdBu',
                             extent=[time_vec[0], time_vec[-1], x_coord_vec[0], x_coord_vec[-1]],
                             vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Time (s)')
    # axes[1].set_ylabel('X-coordinate (km)') # Y-label is shared
    axes[1].set_title(f"{title_prefix} - Prediction")

    # Add a shared colorbar
    fig.colorbar(im_pred, ax=axes[1], shrink=0.75, label='Value')

    fig.tight_layout()
    return fig


if __name__ == '__main__':

   
    assert options.nlayers == len(options.list_D1)


    name_config = f"JeanZayTEnc_RandBounds-"\
        f"{options.model_type}3D-{options.source_orientation}-"\
        f"dv{options.dv}-{options.nlayers}layers-S{options.S_in}-T{options.T_out}-"\
        f"learningrate{str(options.learning_rate).replace('.','p')}-Ntrain{options.Ntrain}-"\
            f"batchsize{options.batch_size}-"
    if options.normalize_source:
        name_config += "-normedsource"
    if options.normalize_traces is not None:
        name_config += "-normedtraces"
    name_config += options.additional_name
    

    wandb.login()

    train_data = GeologyTracesSourceMaskDataset(
    path_data='/lustre/fsn1/projects/rech/xvy/upz57sx/hemews3d/formatted/',
    #path_data='./data/formatted/',
    dir_data=options.dir_data_train,
    S_in=options.S_in,
    S_in_z=options.S_in_z,
    S_out=options.S_out,
    T_out=options.T_out,
    transform_a='normal',
    N=options.Ntrain,
    orientation=options.source_orientation,
    transform_position=[9600, 9600, 9600],
    transform_angle='unit',
    transform_traces=None
    )

    val_data = GeologyTracesSourceMaskDataset(
    path_data='/lustre/fsn1/projects/rech/xvy/upz57sx/hemews3d/formatted/',
    #path_data='./data/formatted/',
    dir_data=options.dir_data_val,
    S_in=options.S_in,
    S_in_z=options.S_in_z,
    S_out=options.S_out,
    T_out=options.T_out,
    transform_a='normal',
    N=options.Nval,
    orientation=options.source_orientation,
    transform_position=[9600, 9600, 9600],
    transform_angle='unit',
    transform_traces=None
    )
    '''
    train_data = GeologyTracesSourceDataset(path_data='/gpfs/workdir/caballerf/perronen/data/formatted/',
                                            #path_data='./data/formatted/',
                                            dir_data=options.dir_data_train, 
                                            S_in=options.S_in, 
                                            S_in_z=options.S_in_z,
                                            S_out=options.S_out, 
                                            T_out=options.T_out, 
                                            transform_a='normal',
                                            N=options.Ntrain, 
                                            orientation=options.source_orientation,
                                            transform_position=None, 
                                            transform_angle='unit', 
                                            transform_traces=None)

    val_data = GeologyTracesSourceDataset(path_data='/gpfs/workdir/caballerf/perronen/data/formatted/',
                                          #path_data='./data/formatted/',
                                          dir_data=options.dir_data_val, 
                                          S_in=options.S_in, 
                                          S_in_z=options.S_in_z,
                                          S_out=options.S_out, 
                                          T_out=options.T_out, 
                                          transform_a='normal',
                                          N=options.Nval, 
                                          orientation=options.source_orientation,
                                          transform_position=None, 
                                          transform_angle='unit', 
                                          transform_traces=None)
    '''
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=options.batch_size, 
                                               shuffle=True, 
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, 
                                             batch_size=options.batch_size, 
                                             shuffle=False, 
                                             num_workers=2)

    model = GeologyModel(options)
    model=model.to(gpu) # Ensure model is on the correct device

    # Lightning Trainer with DDP Strategy
    '''
    checkpoint_callback = ModelCheckpoint(save_last=True, 
                                          filename="model_ckpt-" + name_config + "-{step}-{train_loss:.2f}",
                                          every_n_train_steps=100, 
                                          dirpath=options.dir_logs)
    '''
    
    modelcheckpoint_callback_regular_step_save = ModelCheckpoint(
        dirpath=options.dir_logs,
        filename="model_ckpt-" + name_config + "-{step}-{loss_train:.2f}",
        every_n_train_steps=100,
        save_last=True
    )
    modelcheckpoint_callback_regular_step_save.CHECKPOINT_NAME_LAST = f"last-{name_config}" # customize the name of the last checkpoint
    modelcheckpoint_callback_best_val_save = ModelCheckpoint(
        dirpath=options.dir_logs,
        # Correct the filename format string to use the monitored metric
        filename="model_best_val_ckpt-" + name_config + "-{epoch}-{val_loss:.2f}",
        # Change monitor to the metric actually logged in validation_step
        monitor="val_loss",
        save_top_k=1,
        mode='min',
        save_on_train_epoch_end=False
    )
    
    
    
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=60, 
                                   mode='min')

    # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project='MaskMIFNO', 
        name=name_config,
        config=vars(options), # Pass all parsed options to Wandb config
        save_dir="/lustre/fsn1/projects/rech/xvy/upz57sx/MIFNO_logs/"
    )
    trainer = Trainer(
        max_epochs=options.epochs,
        accelerator='gpu',
        #strategy=DDP(model,device_ids=[idr_torch.local_rank], find_unused_parameters=True), # Use DDP with the local rank
        devices= int(os.environ['SLURM_GPUS_ON_NODE']), # Use the number of GPUs available
        num_nodes=int(os.environ['SLURM_NNODES']),
        #strategy=DDPStrategy(find_unused_parameters=True),
        #strategy='Auto',
        strategy='ddp_find_unused_parameters_true',
        callbacks=[modelcheckpoint_callback_regular_step_save,modelcheckpoint_callback_best_val_save, early_stopping],
        logger=wandb_logger,  
    )
    
    if options.restart_model:
        print("Restarting from checkpoint...")
        # Construct the correct path including the directory and extension
        latest_checkpoint_filename = modelcheckpoint_callback_regular_step_save.CHECKPOINT_NAME_LAST + ".ckpt"
        trainer_fit_ckpt_path = os.path.join(options.dir_logs, latest_checkpoint_filename) # Use os.path.join for robustness

        # Check if the latest checkpoint file exists before passing it
        if not os.path.exists(trainer_fit_ckpt_path):
            print(f"Warning: Checkpoint specified but not found at {trainer_fit_ckpt_path}. Starting training from scratch.")
            trainer_fit_ckpt_path = None # Set to None if file doesn't exist to start fresh
        else:
            print(f"Found checkpoint at: {trainer_fit_ckpt_path}")

    else:
        print("Starting training from scratch (no checkpoint specified).")
        trainer_fit_ckpt_path = None
    
    trainer.fit(model, train_loader, val_loader, ckpt_path=trainer_fit_ckpt_path)

    # --- Plotting after training ---
    print("Training finished. Generating comparison plot...")

    # Load the best checkpoint
    best_model_path = modelcheckpoint_callback_best_val_save.best_model_path
    if best_model_path:
        print(f"Loading best model from: {best_model_path}")
        model = GeologyModel.load_from_checkpoint(best_model_path)
        model.eval()  # Set model to evaluation mode
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) # Ensure model is on the correct device
    else:
        print("Could not find the best model checkpoint. Using the current model state.")
        model.eval()

    # Get a sample batch from the validation loader
    val_batch = next(iter(val_loader))
    a, uE, uN, uZ, s, grid_bounds, norm_cst = val_batch

    # Move data to the model's device
    device = model.device
    a = a.to(device)
    s = s.to(device)
    grid_bounds = grid_bounds.to(device)
    norm_cst = norm_cst.to(device)
    uE_truth = uE.to(device)
    uN_truth = uN.to(device)
    uZ_truth = uZ.to(device)

    # Perform inference
    with torch.no_grad():
        outE, outN, outZ = model(a, s, grid_bounds)

    # Select the first sample and a component (e.g., uE)
    output_sample = outE[0].cpu().numpy()
    truth_sample = uE_truth[0].cpu().numpy()


    print(f"Debug: output_sample shape: {output_sample.shape}, min: {output_sample.min():.4f}, max: {output_sample.max():.4f}, mean: {output_sample.mean():.4f}")
    print(f"Debug: truth_sample shape: {truth_sample.shape}, min: {truth_sample.min():.4f}, max: {truth_sample.max():.4f}, mean: {truth_sample.mean():.4f}")
    if np.isnan(output_sample).any():
        print("Debug: output_sample contains NaNs!")
    if np.isinf(output_sample).any():
        print("Debug: output_sample contains Infs!")

    # Choose a y-index (e.g., middle)
    y_index = options.S_out // 2

    # Extract the 2D slice (x_dim, time_dim)
    output_slice = output_sample[:, y_index, :]
    truth_slice = truth_sample[:, y_index, :]

    print(f"Debug: output_slice for plot shape: {output_slice.shape}, min: {output_slice.min():.4f}, max: {output_slice.max():.4f}, mean: {output_slice.mean():.4f}")
    print(f"Debug: truth_slice for plot shape: {truth_slice.shape}, min: {truth_slice.min():.4f}, max: {truth_slice.max():.4f}, mean: {truth_slice.mean():.4f}")
    if np.isnan(output_slice).any():
        print("Debug: output_slice for plot contains NaNs!")
    if np.isinf(output_slice).any():
        print("Debug: output_slice for plot contains Infs!")

    # Define time and x-coordinate vectors (assuming dt=0.02s and spatial extent 9.6km)
    dt = 0.02
    time_vec = np.linspace(0, (options.T_out - 1) * dt, options.T_out)
    x_coord_vec = np.linspace(0, 9600, options.S_out)

    max_abs_val_check = max(np.abs(truth_slice).max(), np.abs(output_slice).max()) # Recalculate for check
    vmin_check = -max_abs_val_check
    vmax_check = max_abs_val_check
    print(f"Debug: For plotting - max_abs_val: {max_abs_val_check:.4f}, vmin: {vmin_check:.4f}, vmax: {vmax_check:.4f}")

    # --- Create and log the comparison plot ---
    fig_comparison_final = plot_comparison_time_vs_x(
        truth_slice, output_slice, time_vec, x_coord_vec,
        title_prefix=f"Final Comparison (uE) at y-index {y_index}"
    )
    # Log the combined plot to Wandb
    if wandb_logger:
        wandb_logger.experiment.log({"final_plot/comparison_E": wandb.Image(fig_comparison_final)})
   
    #wandb.log({"final_plot/comparison_E": wandb.Image(fig_comparison_final)})
    plt.close(fig_comparison_final) # Close the figure after logging
    '''
    # Plot ground truth and log to Wandb
    fig_truth_final = plot_time_vs_x(truth_slice, time_vec, x_coord_vec,
                               title=f"Final - Ground Truth (uE) at y-index {y_index}")
    wandb.log({"final_plot/ground_truth_E": wandb.Image(fig_truth_final)})
    plt.close(fig_truth_final)

    # Plot model output and log to Wandb
    fig_pred_final = plot_time_vs_x(output_slice, time_vec, x_coord_vec,
                               title=f"Final - Model Output (outE) at y-index {y_index}")
    wandb.log({"final_plot/prediction_E": wandb.Image(fig_pred_final)})
    plt.close(fig_pred_final)
    '''
    print("Plotting complete.")
    #wandb.finish() # Ensure Wandb run finishes after logging plots

