import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet
from engine import train_one_epoch, val_one_epoch, test_one_epoch
import os
import sys
from utils import get_logger, set_seed, log_config_info, cal_params_flops, get_optimizer, get_scheduler
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt  # Grafik çizimi için eklendi
import wandb 
from configs.config_setting import setting_config
from dataclasses import asdict

# Eğer setting_config bir dataclass ise:
# from dataclasses import asdict
# config_dict = asdict(setting_config)

config_dict = {
    k: getattr(setting_config, k)
    for k in dir(setting_config)
    if not k.startswith('__') and not callable(getattr(setting_config, k))
}

def main(config):

    wandb.init(project="data-science-project", config=config_dict)

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size, 
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()
    else:
        raise Exception('network is not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Preparing loss, optimizer, scheduler and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = float('inf')
    start_epoch = 1
    min_epoch = 1
    early_stop_count = 0
    early_stop_patience = getattr(config, "early_stop_patience", 30)  # erken durdurma için sabır sayısı, default 10 epoch

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, last_loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        log_info = f'Resuming model from {resume_model}. Resume epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {last_loss:.4f}'
        logger.info(log_info)

    step = 0
    train_losses = []  # her epoch için train loss
    val_losses = []    # her epoch için validation loss

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        # train_one_epoch artık (step, epoch_train_loss) döndürüyor
        step, epoch_train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        epoch_val_loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "lr": optimizer.state_dict()['param_groups'][0]['lr']
        })

        # En iyi validation loss kontrolü ve erken durdurma sayacının güncellenmesi
        if epoch_val_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = epoch_val_loss
            min_epoch = epoch
            early_stop_count = 0
        else:
            early_stop_count += 1

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': epoch_val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, 'latest.pth')
        )

        # # Erken durdurma kontrolü
        # if early_stop_count >= early_stop_patience:
        #     log_info = f"Early stopping triggered at epoch {epoch}. No improvement in validation loss for {early_stop_patience} consecutive epochs."
        #     print(log_info)
        #     logger.info(log_info)
        #     break

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )

    # Eğitim sonunda train ve validation loss eğrilerini çiz ve PNG olarak kaydet
    plt.figure()
    epochs_range = list(range(start_epoch, start_epoch + len(train_losses)))
    plt.plot(epochs_range, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs_range, val_losses, marker='o', label='Validation Loss')
    plt.title('Train and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(outputs, 'loss_plot.png')
    plt.savefig(plot_path)
    # plt.show()
    print(f'Loss plot saved to {plot_path}')
    
    wandb.finish()

if __name__ == '__main__':
    config = setting_config
    main(config)
