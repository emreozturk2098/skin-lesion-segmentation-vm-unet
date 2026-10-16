import os
import torch
import shutil
import optuna
import csv
import gc

def objective(trial):
    # Her trial için yeni bir konfigürasyon örneği oluşturuluyor.
    from configs.config_setting import setting_config
    config = setting_config()

    # GPU belleğini etkilemeyecek hiperparametreler ayarlanıyor.
    config.lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    config.weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-2)
    config.model_config['drop_path_rate'] = trial.suggest_uniform('drop_path_rate', 0.1, 0.5)
    config.early_stop_patience = trial.suggest_int('early_stop_patience', 5, 15)

    # Her trial için farklı bir çalışma dizini oluşturuluyor.
    config.work_dir = f"results/optuna_trial_{trial.number}/"

    try:
        # Eğitim işlemi başlatılıyor.
        from train import main
        main(config)
    except RuntimeError as e:
        # CUDA out-of-memory hatasını yakalayıp, trial'ı başarısız (infinite loss) olarak işaretliyoruz.
        if "CUDA out of memory" in str(e):
            print(f"Trial {trial.number} CUDA OOM hatası verdi. Bellek temizleniyor...")
            torch.cuda.empty_cache()
            gc.collect()
            return float('inf')
        else:
            raise e

    # Eğitim sonrası, en iyi validasyon kaybını almak için kontrol noktası yükleniyor.
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    latest_path = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=torch.device('cpu'))
        best_loss = ckpt.get('min_loss', float('inf'))
    else:
        best_loss = float('inf')

    # Trial çalışma dizini siliniyor.
    shutil.rmtree(config.work_dir, ignore_errors=True)

    # GPU belleği temizleniyor.
    torch.cuda.empty_cache()
    gc.collect()

    return best_loss

if __name__ == '__main__':
    # Hiperparametre optimizasyonu için Optuna çalışması oluşturuluyor (minimize yönünde).
    study = optuna.create_study(direction='minimize')
    # Örnek sayısı 10 olarak belirleniyor.
    study.optimize(objective, n_trials=10)

    # En iyi trial bilgileri yazdırılıyor.
    best_trial = study.best_trial
    print("Best Trial:")
    print(f"  Value: {best_trial.value}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Tüm denemelerin sonuçlarını içeren CSV raporu oluşturuluyor.
    report_file = "optuna_study_report.csv"
    with open(report_file, mode='w', newline='') as csv_file:
        fieldnames = ['trial_number', 'value'] + list(best_trial.params.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            row = {'trial_number': trial.number, 'value': trial.value}
            for key in best_trial.params.keys():
                row[key] = trial.params.get(key, None)
            writer.writerow(row)

    print(f"CSV report saved to {report_file}")
