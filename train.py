import torch as t
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import itertools
import sys

from data import ChallengeDataset
from trainer import Trainer
import model


# --------------------------------------------------
# Utility
# --------------------------------------------------
def set_seed(seed=42):
    t.manual_seed(seed)
    np.random.seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def export_onnx(model_instance, save_path, device):
    """Esporta il modello in formato ONNX per la sottomissione"""
    model_instance.eval()
    # Dummy input: (Batch, Canali, Altezza, Larghezza)
    dummy_input = t.randn(1, 3, 300, 300).to(device)

    print(f"Esportazione modello ONNX in {save_path}...")
    try:
        t.onnx.export(
            model_instance,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("✅ Esportazione ONNX completata con successo.")
    except Exception as e:
        print(f"❌ Errore durante l'esportazione ONNX: {e}")


# --------------------------------------------------
# Training + evaluation function (Tuning)
# --------------------------------------------------
def train_and_evaluate(config, train_df, val_df, device, use_gpu, epochs=50, save_name=None):
    set_seed(42)

    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    optimizer_name = config["optimizer"]

    train_dataset = ChallengeDataset(train_df, mode='train')
    val_dataset = ChallengeDataset(val_df, mode='val')

    train_loader = t.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = t.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    resnet_model = model.ResNet().to(device)
    criterion = t.nn.BCELoss()

    if optimizer_name == "adam":
        optimizer = t.optim.Adam(resnet_model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = t.optim.SGD(resnet_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("Unknown optimizer")

    trainer = Trainer(
        model=resnet_model,
        crit=criterion,
        optim=optimizer,
        train_dl=train_loader,
        val_test_dl=val_loader,
        cuda=use_gpu,
        early_stopping_patience=15
    )

    train_losses, val_losses = trainer.fit(epochs=epochs)

    if save_name:
        # 1. Salva il Checkpoint PyTorch (.ckp)
        ckpt_path = f'checkpoints/{save_name}.ckp'
        t.save(resnet_model.state_dict(), ckpt_path)
        print(f"✅ Checkpoint salvato in {ckpt_path}")

        # 2. Esporta la versione ONNX (.onnx)
        onnx_path = f'checkpoints/{save_name}.onnx'
        export_onnx(resnet_model, onnx_path, device)

    return min(val_losses), train_losses, val_losses


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    final_model_name = "model_final"
    final_ckpt_path = checkpoint_dir / f"{final_model_name}.ckp"
    final_onnx_path = checkpoint_dir / f"{final_model_name}.onnx"

    is_refine_mode = "refine_best" in sys.argv

    print('Caricamento dati...')
    df = pd.read_csv('data.csv', sep=';')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    if t.cuda.is_available():
        device, use_gpu = 'cuda', True
    elif t.backends.mps.is_available():
        device, use_gpu = 'mps', True
    else:
        device, use_gpu = 'cpu', False

    print(f'Using device: {device.upper()}')

    if not is_refine_mode:
        # --- MODALITÀ TUNING ---
        print("\n🔍 MODALITÀ TUNING ATTIVATA\n")
        param_grid = {
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "batch_size": [16, 32],
            "optimizer": ["adam", "sgd"]
        }

        results = []
        for values in itertools.product(*param_grid.values()):
            config = dict(zip(param_grid.keys(), values))
            print(f"Testing: {config}")
            val_loss, _, _ = train_and_evaluate(config, train_df, val_df, device, use_gpu, epochs=30)
            results.append({**config, "val_loss": val_loss})

        results_df = pd.DataFrame(results).sort_values(by='val_loss')
        results_df.to_csv("hparam_results.csv", index=False)
        print("\n🏆 Risultati salvati in hparam_results.csv")

    else:
        # --- MODALITÀ REFINE (Carica, Allena, Esporta) ---
        print("\n🚀 MODALITÀ REFINE & EXPORT")

        # Iperparametri con regolarizzazione aumentata
        best_config = {
            "learning_rate": 3e-5,      # LR più basso
            "batch_size": 32,            # Batch size maggiore per stabilità
            "optimizer": "adam",
            "weight_decay": 5e-4         # Weight decay più forte
        }

        # Inizializza modello
        resnet_model = model.ResNet().to(device)

        # Caricamento Checkpoint per Warm Start
        if final_ckpt_path.exists():
            print(f"📦 Checkpoint trovato: {final_ckpt_path}. Caricamento pesi...")
            resnet_model.load_state_dict(t.load(final_ckpt_path, map_location=device))
        else:
            print(f"⚠️ Nessun checkpoint trovato. Inizio da zero.")

        # Setup dati per il Refinement
        train_dataset = ChallengeDataset(train_df, mode='train')
        val_dataset = ChallengeDataset(val_df, mode='val')
        train_loader = t.utils.data.DataLoader(
            train_dataset,
            batch_size=best_config["batch_size"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True  # Evita batch incompleti
        )
        val_loader = t.utils.data.DataLoader(
            val_dataset,
            batch_size=best_config["batch_size"],
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        # Optimizer con weight decay più forte
        optimizer = t.optim.Adam(
            resnet_model.parameters(),
            lr=best_config["learning_rate"],
            weight_decay=best_config["weight_decay"]
        )

        trainer = Trainer(
            model=resnet_model,
            crit=t.nn.BCELoss(),
            optim=optimizer,
            train_dl=train_loader,
            val_test_dl=val_loader,
            cuda=use_gpu,
            early_stopping_patience=8  # Stop più veloce se val loss non migliora
        )

        print("Avvio addestramento di raffinamento (anti-overfitting)...")
        train_losses, val_losses = trainer.fit(epochs=30)  # Meno epoche

        # Salvataggio finale doppio
        print("\nSalvataggio in corso...")
        t.save(resnet_model.state_dict(), final_ckpt_path)
        export_onnx(resnet_model, str(final_onnx_path), device)

        # Calcolo metriche F1/Precision/Recall
        print("\n📊 Calcolo metriche finali sul validation set...")
        resnet_model.eval()
        preds_all, labels_all = [], []
        #with t.no_grad():
        #    for x, y in val_loader:
        #        out = resnet_model(x.to(device))
        #        preds_all.extend((out > 0.5).float().cpu().numpy())
        #        labels_all.extend(y.numpy())

        #f1 = f1_score(labels_all, preds_all)
        print(f"--- REPORT FINALE ---")
        #print(f"F1 Score:  {f1:.4f}")
        #print(f"Precision: {precision_score(labels_all, preds_all):.4f}")
        #print(f"Recall:    {recall_score(labels_all, preds_all):.4f}")
        print(f"----------------------")

        # Plot finale
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.yscale('log')
        plt.title(f'Final Refinement (Best Val Loss: {min(val_losses):.6f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('refine_losses.png')
        print(f"Plot salvato in refine_losses.png")