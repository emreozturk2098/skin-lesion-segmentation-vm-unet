import os
import sys
import random
import warnings
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wandb
from tensorboardX import SummaryWriter

from datasets.dataset import NPY_datasets
from engine import train_one_epoch, val_one_epoch, test_one_epoch
from models.vmunet.vmunet import VMUNet
from utils import (
    get_logger, set_seed, log_config_info,
    cal_params_flops, get_optimizer, get_scheduler
)
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")

# Config'u dict haline getirme (wandb için)
config_dict = {
    k: getattr(setting_config, k)
    for k in dir(setting_config)
    if not k.startswith('__') and not callable(getattr(setting_config, k))
}

def main(config):
    wandb.init(project="data-science-project", config=config_dict)

    # Logger ve dizin ayarları
    sys.path.append(config.work_dir + '/')
    log_dir        = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs        = os.path.join(config.work_dir, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    log_config_info(config, logger)

    print('# GPU init #')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('# Dataset hazırlama #')
    train_dataset = NPY_datasets(config.data_path, config, split='train')
    train_loader  = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers
    )

    val_dataset = NPY_datasets(config.data_path, config, split='val')
    val_loader  = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    test_dataset = NPY_datasets(config.data_path, config, split='test')
    test_loader  = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers
    )

    print('# Model hazırlama #')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes     = model_cfg['num_classes'],
            input_channels  = model_cfg['input_channels'],
            depths          = model_cfg['depths'],
            depths_decoder  = model_cfg['depths_decoder'],
            drop_path_rate  = model_cfg['drop_path_rate'],
            load_ckpt_path  = model_cfg['load_ckpt_path'],
        )
        model.load_from()
    else:
        raise ValueError('Yanlış network seçimi!')
    model = model.cuda()
    cal_params_flops(model, 256, logger)

    print('# Loss, optimizer, scheduler #')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    min_loss         = float('inf')
    start_epoch      = 1
    early_stop_count = 0
    step             = 0   # başlatıyoruz

    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(resume_model):
        chk = torch.load(resume_model, map_location='cpu')
        model.load_state_dict(chk['model_state_dict'])
        optimizer.load_state_dict(chk['optimizer_state_dict'])
        scheduler.load_state_dict(chk['scheduler_state_dict'])
        saved_ep     = chk['epoch']
        step         = chk.get('step', 0)
        start_epoch  = saved_ep + 1
        min_loss     = chk['min_loss']
        logger.info(f"Resume: epoch {saved_ep}, min_loss {min_loss:.4f}")

    train_losses, val_losses = [], []

    print('# Eğitim başlıyor #')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        # train_one_epoch'tan dönen step ve loss
        step, train_loss = train_one_epoch(
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

        val_loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # wandb'e log
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })

        # En iyi modeli kaydet
        if val_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        # latest.pth güncelle
        torch.save({
            'epoch': epoch,
            'step': step,
            'min_loss': min_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, resume_model)

    # Test aşaması
    best_pth = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_pth):
        print('# Test başlıyor #')
        model.load_state_dict(torch.load(best_pth, map_location='cpu'))
        test_loss = test_one_epoch(
            test_loader, model, criterion,
            logger, config
        )
        print(f"Test Loss: {test_loss:.4f}")

    # Loss eğrisi çizimi ve kaydı
    plt.figure()
    epochs = list(range(start_epoch, start_epoch + len(train_losses)))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses,   marker='o', label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(outputs, 'loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

    writer.close()
    wandb.finish()

if __name__ == '__main__':
    main(setting_config)

