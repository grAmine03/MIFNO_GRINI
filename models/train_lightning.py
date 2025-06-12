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

from utils_models import get_device, get_batch_size, loss_criterion, RunningAverage
from ffno_model import FFNO_3D
from mifno_model import MIFNO_3D
from dataloaders import GeologyTracesSourceDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="MIFNO", help="Architecture used: MIFNO or F-FNO")
parser.add_argument('--S_in', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('--S_in_z', type=int, default=32, help="Size of the spatial input grid")
parser.add_argument('--S_out', type=int, default=32, help="Size of the spatial output grid")
parser.add_argument('--T_out', type=int, default=320, help="Number of time steps")
parser.add_argument('--nlayers', type=int, default=16, help="Number of layers")
parser.add_argument('--branching_index', type=int, default=4, help="Index of the first FFNO block seeing the source")
parser.add_argument('--dv', type=int, help = "Number of channels")
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
parser.add_argument('--dir_data_train', type=str, nargs='+', default=['../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_train'], help="Name of folders with training data")
parser.add_argument('--dir_data_val', type=str, nargs='+', default=['../data/formatted/HEMEWS3D_S32_Z32_T320_fmax5_rot0_val'], help="Name of folders with training data")
parser.add_argument('--dir_logs', type=str, default='../logs/', help="Path to folder to store loss and models")
parser.add_argument('--additional_name', type=str, default="", help="string to add to the configuration name for saved outputs")
parser.add_argument('--restart_model', type=str, default="", help="Path to the model to use as initialization")
parser.add_argument('--start_epoch', type=int, default=0, help="Epoch to start, >0 if initializing with a trained model")
parser.add_argument('--seed', type=int, default=0, help="Seed to initialize pytorch")
options = parser.parse_args()


# Set seeds and deterministic behavior
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        self.loss_weights = options.loss_weights
        self.loss_criterion = loss_criterion

    def forward(self, a, s):
        return self.model(a, s)

    def training_step(self, batch, batch_idx):
        a, uE, uN, uZ, s = batch
        outE, outN, outZ = self(a, s)
        loss_rel = self.loss_criterion((outE, outN, outZ), (uE, uN, uZ), self.loss_weights, relative=True)
        self.log('train_loss', loss_rel, on_step=False, on_epoch=True, prog_bar=True)
        return loss_rel

    def validation_step(self, batch, batch_idx):
        a, uE, uN, uZ, s = batch
        outE, outN, outZ = self(a, s)
        loss_rel = self.loss_criterion((outE, outN, outZ), (uE, uN, uZ), self.loss_weights, relative=True)
        self.log('val_loss', loss_rel, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=options.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


if __name__ == '__main__':

    #assert options['nlayers'] == len(options['list_D1'])
    assert options.nlayers == len(options.list_D1)


    name_config = f"{options.model_type}3D-{options.source_orientation}-"\
        f"dv{options.dv}-{options.nlayers}layers-S{options.S_in}-T{options.T_out}-"\
        f"learningrate{str(options.learning_rate).replace('.','p')}-Ntrain{options.Ntrain}-"\
            f"batchsize{options.batch_size}"
    if options.normalize_source:
        name_config += "-normedsource"
    if options.normalize_traces is not None:
        name_config += "-normedtraces"
    name_config += options.additional_name

    train_data = GeologyTracesSourceDataset(options.dir_data_train, 
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

    val_data = GeologyTracesSourceDataset(options.dir_data_val, 
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


    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=options.batch_size, 
                                               shuffle=True, 
                                               num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, 
                                             batch_size=options.batch_size, 
                                             shuffle=False, 
                                             num_workers=2)

    model = GeologyModel(options)

    # Lightning Trainer with DDP Strategy
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                          save_top_k=1, 
                                          mode='min', 
                                          dirpath=options['dir_logs'])
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=60, 
                                   mode='min')

    trainer = Trainer(
        max_epochs=options['epochs'],
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(),
        callbacks=[checkpoint_callback, early_stopping]
    )

    trainer.fit(model, train_loader, val_loader)
