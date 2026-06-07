# Rapport d'Audit Approfondi — NaylisGDN (Hopper/1B)

Cet audit porte sur l'architecture, la stabilité et la performance du modèle hybride GDN-2 + Transformer, avec un focus sur l'entraînement à l'échelle (1B+ paramètres) en FP8/BF16 avec FSDP.

---

## 1. Bugs Probables et Erreurs Fatales

### 🔴 Distribution des données FSDP (train.py)
**Bug :** L'entraînement utilise `torch.utils.data.SequentialSampler` dans un contexte FSDP.
**Impact :** Toutes les instances (ranks) liront **exactement les mêmes données**. En FSDP 8-GPU, le modèle verra 8 fois moins de données uniques et gaspillera 87.5% de la puissance de calcul.
**Correction :** Utiliser `torch.utils.data.distributed.DistributedSampler`.

### 🔴 Checkpointing FSDP Corrompu (train.py)
**Bug :** `CheckpointManager.save` appelle `torch.save` sans condition de rank.
**Impact :** Les 8 GPUs vont tenter d'écrire simultanément dans le même fichier `.pt`, provoquant des corruptions de fichiers ou des crashes IO. De plus, `model.state_dict()` sur un modèle FSDP sans configuration spécifique ne retourne que les shards locaux.
**Correction :** Utiliser un context manager `FSDP.state_dict_type` et ne sauvegarder que sur le `rank == 0`.

### 🔴 Erreur d'initialisation `soft_cap` (naylisgdn/model.py)
**Bug :** `assert 0 < soft_cap <= 100` (Ligne 72).
**Impact :** Si `soft_cap` est mis à `0` (valeur par défaut dans certains TOML pour dire "désactivé"), le modèle crash à l'init.
**Correction :** Autoriser `soft_cap is None` ou `0` et adapter l'assertion.

### 🟠 Variable non définie en Gradient Checkpointing (naylisgdn/model.py)
**Bug :** Dans `NaylisGDN.forward`, si `use_gc` est activé, la variable `aux` n'est pas assignée dans le bloc `if use_gc`, mais est utilisée plus bas : `total_aux_loss = total_aux_loss + aux` (Ligne 176).
**Impact :** `UnboundLocalError` immédiat lors de l'entraînement avec gradient checkpointing.
**Correction :** Initialiser `aux = 0` dans le bloc `if use_gc`.

---

## 2. Incohérences README / Config / Code

- **Dtype des données :** Le README et `train.py` mentionnent `uint32`, mais `config/1B.toml` contient `token_dtype = "uint16"`. Pour un vocab de 49k, `uint16` suffit, mais l'incohérence peut mener à des erreurs de lecture `memmap` si les fichiers sont générés en 32-bits.
- **Paramètres GDN 1B :** Le TOML 1B définit `gdn_head_dim=128` avec `16` têtes (soit 2048 dimensions projetées) pour un `embed_dim` de 1408. Ce n'est pas un bug technique, mais c'est une projection "expandante" inhabituelle par rapport aux standards GDN-2.
- **Repo HF :** `train.py` a en dur `_HF_DATA_REPO = "silyan/Naylis1-1.3B"`, tandis que le TOML 1B pointe vers `silyan/Naylis1-1B`. Risque de télécharger les mauvais chunks.

---

## 3. Risques Silencieux en ML (Robustesse)

### ⚠️ Positions RoPE en Sequence Packing (CRITIQUE)
**Problème :** `RotaryPositionalEmbedding` utilise un `arange(seq_len)` continu.
**Impact :** Dans une séquence packée (ex: Doc A de 4k + Doc B de 4k), le Doc B commence avec des indices de position 4001 à 8000 au lieu de 0 à 4000. Cela dégrade fortement la performance sur les documents courts situés en fin de pack.
**Test :** `debugue_test/test_rope_packing.py` a confirmé que les positions ne sont pas reset.

### ⚠️ Fuite d'Attention (Attention Leakage)
**Problème :** En cas de fallback `SDPA` (si Flash Attention n'est pas dispo), `is_causal=True` est utilisé.
**Impact :** `SDPA` ne connaît pas les `cu_seqlens`. Le Doc B pourra "voir" le Doc A à travers l'attention causale standard.
**Test :** `debugue_test/test_attention_leakage.py` a confirmé la fuite.

---

## 4. Points de Performance Suspects

- **einops.repeat dans GDNBlock :** L'utilisation de `einops.repeat` dans la boucle de forward (Ligne 198) pour matcher les têtes QK et V peut ralentir le kernel Triton si celui-ci n'est pas déjà optimisé pour le GQA interne au GDN.
- **Calcul log-decay :** Le calcul `_compute_log_decay` force un cast en `.float()` (fp32). C'est nécessaire pour la stabilité mais coûteux sur Hopper si fait trop souvent hors du kernel.
- **CPU Fallback :** Les kernels Triton GDN-2 n'ont pas de fallback optimisé en C++/CUDA pur, seulement un fallback PyTorch très lent (`_gdn2_torch`). Sur un nœud sans Triton fonctionnel, l'entraînement sera 50x plus lent.

---

## 5. Fichiers à Surveiller en Priorité

1.  `naylisgdn/rope.py` : Doit impérativement supporter le reset des positions via `cu_seqlens`.
2.  `naylisgdn/attention.py` : Doit gérer le masquage par bloc pour le packing en mode fallback SDPA.
3.  `train.py` : La logique de reprise mid-chunk est complexe et repose sur des calculs d'indices (`skip_batches`) qui doivent être parfaitement synchronisés avec le `DistributedSampler`.

---

## 6. Suggestions Précises de Corrections

1.  **RoPE :** Modifier `RotaryPositionalEmbedding.forward` pour accepter un tenseur `position_ids` optionnel, généré dans `train.py` ou `model.py` à partir des `cu_seqlens`.
2.  **Attention :** Dans `MultiHeadAttention.forward`, si `cu_seqlens` est fourni mais `use_varlen` est False (fallback), générer un masque 2D 4D `[B, 1, T, T]` bloquant les attentions inter-documents.
3.  **FSDP :**
    - Ajouter `from torch.utils.data.distributed import DistributedSampler`.
    - Envelopper la sauvegarde dans `if _LOCAL_RANK == 0:`.
    - Utiliser `FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT)`.
4.  **Model :** Initialiser `aux = torch.zeros(1, ...)` au début de `forward` pour éviter les erreurs de variable non définie avec MoE/GC.
5.  **Config :** Normaliser les types (toujours `uint32` pour le 1B+ pour éviter tout overflow d'indexation).

---
*Rapport généré par Jules — Audit interne NaylisGDN.*
