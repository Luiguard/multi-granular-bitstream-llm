#!/usr/bin/env python3
"""
Warm Vocabulary Expansion: 16-Bit (65,536) -> 18-Bit (262,144) Checkpoint Adapter.
Preserves 100% of existing learned weights from Step 1,640+ and warm-initializes
new token slots (65,536 to 262,143).
"""

import json
import os
import shutil
import sys
import torch

sys.path.insert(0, "/home/benjamin/Bilder")

OLD_VOCAB_SIZE = 65536
NEW_VOCAB_SIZE = 262144
RANK = 64
CHECKPOINT_DIR = "/home/benjamin/Bilder/checkpoints"
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "7b_checkpoint_latest.pt")
BACKUP_CKPT = os.path.join(CHECKPOINT_DIR, "7b_checkpoint_16bit_step_1640_backup.pt")


def expand_checkpoint_to_18bit():
    print("=" * 80)
    print("🔄 WARM VOCABULARY EXPANSION: 16-BIT (65.536) -> 18-BIT (262.144)")
    print("=" * 80)

    if not os.path.exists(LATEST_CKPT):
        print(f"❌ Checkpoint {LATEST_CKPT} nicht gefunden!")
        return

    # 1. Erstelle Sicherheitskopie des bisherigen 16-Bit Checkpoints
    print(f"  💾 Erstelle Sicherheits-Backup: {BACKUP_CKPT}...")
    shutil.copyfile(LATEST_CKPT, BACKUP_CKPT)

    # 2. Lade 16-Bit Checkpoint via mmap
    print(f"  📥 Lade 16-Bit Checkpoint: {LATEST_CKPT}...")
    ckpt = torch.load(LATEST_CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt["model_state_dict"] if (isinstance(ckpt, dict) and "model_state_dict" in ckpt) else ckpt

    step = ckpt.get("step", 1640) if isinstance(ckpt, dict) else 1640
    tokens = ckpt.get("tokens_processed", step * 7168) if isinstance(ckpt, dict) else step * 7168

    print(f"  • Bisheriger Trainingsstand: Step {step:,} ({tokens:,} Tokens)")

    # 3. Erweitere E_vocab (Embedding-Tabelle)
    old_e_vocab = sd["E_vocab.weight"]
    print(f"  • Altes E_vocab Shape: {old_e_vocab.shape}")
    
    if old_e_vocab.shape[0] < NEW_VOCAB_SIZE:
        new_e_vocab = torch.empty((NEW_VOCAB_SIZE, RANK), dtype=old_e_vocab.dtype)
        # 100% exakte Übernahme aller bisher trainierten Gewichte
        new_e_vocab[:OLD_VOCAB_SIZE, :] = old_e_vocab
        
        # Warm-Initialisierung der neuen 196.608 Slots mit berechnetem Mittelwert + kleinem Rauschen
        mean_vec = old_e_vocab.mean(dim=0, keepdim=True)
        std_vec = old_e_vocab.std(dim=0, keepdim=True) * 0.1
        noise = torch.randn((NEW_VOCAB_SIZE - OLD_VOCAB_SIZE, RANK), dtype=old_e_vocab.dtype) * std_vec
        new_e_vocab[OLD_VOCAB_SIZE:, :] = mean_vec + noise
        
        sd["E_vocab.weight"] = new_e_vocab
        print(f"  ✅ Neues E_vocab Shape: {new_e_vocab.shape} (100% Gewichts-Erhalt für [0..65535])")

    # 4. Erweitere head_out (Output-Head-Tabelle)
    old_head_out = sd["head_out.weight"]
    print(f"  • Altes head_out Shape: {old_head_out.shape}")
    
    if old_head_out.shape[0] < NEW_VOCAB_SIZE:
        new_head_out = torch.empty((NEW_VOCAB_SIZE, RANK), dtype=old_head_out.dtype)
        new_head_out[:OLD_VOCAB_SIZE, :] = old_head_out
        
        head_mean = old_head_out.mean(dim=0, keepdim=True)
        head_std = old_head_out.std(dim=0, keepdim=True) * 0.1
        head_noise = torch.randn((NEW_VOCAB_SIZE - OLD_VOCAB_SIZE, RANK), dtype=old_head_out.dtype) * head_std
        new_head_out[OLD_VOCAB_SIZE:, :] = head_mean + head_noise
        
        sd["head_out.weight"] = new_head_out
        print(f"  ✅ Neues head_out Shape: {new_head_out.shape} (100% Gewichts-Erhalt für [0..65535])")

    # 5. Speichere erweiterten 18-Bit Checkpoint atomar
    ckpt["model_state_dict"] = sd
    temp_ckpt = LATEST_CKPT + ".tmp"
    torch.save(ckpt, temp_ckpt)
    os.replace(temp_ckpt, LATEST_CKPT)

    print(f"  💾 18-Bit Checkpoint atomar gespeichert -> {LATEST_CKPT}")
    print("  🎉 Checkpoint erfolgreich auf 18-Bit (262.144 Tokens) skaliert!")


if __name__ == "__main__":
    expand_checkpoint_to_18bit()
