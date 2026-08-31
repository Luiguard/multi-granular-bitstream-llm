#!/usr/bin/env python3
"""Specialized Ingestion Pipeline for Operating Systems Engineering, Linux Distributions,
macOS (Apple Silicon/UNIX), Windows 11/WSL2, Android (AOSP/Kernel), and iOS Architecture.

Covers:
1. Linux Ecosystem & All Major Distributions:
   - Debian / Ubuntu / Pop!_OS: apt, dpkg, PPA, CUDA drivers, developer setups.
   - Arch Linux / Manjaro / EndeavourOS: pacman, AUR, rolling release, bleeding edge.
   - Fedora / RHEL / Rocky: dnf, rpm, SELinux, enterprise stability, Wayland.
   - openSUSE (Tumbleweed/Leap): zypper, YaST, Btrfs + Snapper snapshots.
   - NixOS: Declarative configuration.nix, atomic rollbacks, isolated build environments.
   - Alpine & Void: musl libc, apk, minimal footprint.
   - Linux Kernel Internals: Systemd, cgroups v2, namespaces, eBPF, Wayland, PipeWire, ZFS/Btrfs/ext4.
2. macOS & Apple Silicon (Darwin/UNIX):
   - XNU Hybrid Kernel, Mach IPC, BSD POSIX, APFS, SIP (System Integrity Protection).
   - Apple Silicon Unified Memory Architecture (UMA): MLX, Metal Performance Shaders for local LLMs.
3. Microsoft Windows 11 & Windows Server:
   - NT Kernel, Win32/WinRT, NTFS/ReFS, WSL2 (Hyper-V Linux with direct CUDA GPU passthrough), DirectX 12.
4. Mobile Operating Systems (Android & iOS):
   - Android: AOSP, Linux Kernel, ART runtime, Binder IPC, GrapheneOS privacy, Termux.
   - iOS: XNU, App Sandbox, Secure Enclave, Swift/SwiftUI, Jetsam memory management.
5. Interactive OS Decision Matrix:
   - Tailored matching based on developer needs, gaming, AI development, privacy, and creative workflows.

Appends directly to data/cyber_web_knowledge/shards/ without overwriting existing shards.
"""

import os
import sys
import time
import glob
import signal
from typing import List, Dict, Any

from datasets import load_dataset
from pipeline.vocabulary import MultiGranularVocabulary
from pipeline.bitstream import BitstreamEncoder
from pipeline.tokenizer import ViterbiTokenizer

STOP_REQUESTED = False

def handle_signal(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠️ Graceful Shutdown angefordert, beende nach aktuellem Shard...", flush=True)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


OS_COMPREHENSIVE_KNOWLEDGE = [
    # 1. Linux Distribution Deep Dive & Kernel Architecture
    r"""# Der ultimative Linux-Distributions- & Systemarchitektur-Leitfaden:

## 1. Übersicht der Distributions-Familien & Paketmanager
1. **Debian-Familie (Debian, Ubuntu, Linux Mint, Pop!_OS)**:
   - Paketmanagement: `apt`, `dpkg`, `.deb`.
   - Stärken: Größte Software-Kompatibilität, Standard für KI/Deep Learning (NVIDIA CUDA, PyTorch, Docker), extrem stabil.
   - Pop!_OS: Exzellente NVIDIA-Treiber-Integration und natives Tiling Window Management (COSMIC).
2. **Arch-Familie (Arch Linux, Manjaro, EndeavourOS)**:
   - Paketmanagement: `pacman` und das Arch User Repository (`AUR`).
   - Stärken: Rolling Release (stets neuester Linux-Kernel & Mesa-Treiber), Do-It-Yourself Philosophie, maximale Kontrolle.
3. **Red Hat / Fedora-Familie (Fedora Workstation, RHEL, AlmaLinux, Rocky Linux)**:
   - Paketmanagement: `dnf`, `rpm`.
   - Stärken: Modernste Upstream-Technologien (Wayland, PipeWire, Btrfs), strikte SELinux-Sicherheit, Enterprise-Standard.
4. **openSUSE (Tumbleweed Rolling / Leap Stable)**:
   - Paketmanagement: `zypper`, Systemverwaltung über `YaST`.
   - Stärken: Automatisierte Btrfs-Dateisystem-Snapshots mit `Snapper` (bei fehlerhaftem Update bootet man mit einem Tastendruck in den vorherigen Zustand).
5. **Deklarative Systeme (NixOS)**:
   - Konfiguration über eine zentrale `/etc/nixos/configuration.nix`.
   - Stärken: Vollkommen reproduzierbare Systeme, atomare Upgrades und Rollbacks, zustandsloses OS.

## 2. Linux Kernel Subsysteme & Low-Level
- **Init-System & Daemons**: `systemd` (Units, Targets, `journalctl`, `systemctl`), cgroups v2 für Container-Ressourcenlimitierung.
- **Grafik & Display-Server**: Wayland (Sicherheit durch Client-Isolation) ersetzt das veraltete X11.
- **Audio-Architektur**: PipeWire mit niedriger Latenz für Pro-Audio und Screensharing.""",

    # 2. macOS vs Windows vs Linux vs Android vs iOS Decision Matrix
    r"""# Betriebssystem-Entscheidungsmatrix: Welches System passt zu welchem Profil?

## 1. Deep Learning, KI-Forschung & High-Performance Computing
- **Empfehlung 1: Linux (Ubuntu 24.04 LTS / Pop!_OS)**:
  - Warum: Native NVIDIA CUDA-Beschleunigung ohne Virtualisierungs-Overhead, direkter Zugriff auf GPU-VRAM, Docker mit `nvidia-container-toolkit`.
- **Empfehlung 2: macOS (Apple Silicon M-Serie mit 64GB - 128GB Unified Memory)**:
  - Warum: Riesiger gemeinsamer Arbeitsspeicher (UMA) erlaubt das lokale Laden riesiger 70B+ LLMs im VRAM via Apple MLX / Metal Performance Shaders bei extrem geringem Stromverbrauch (Akkulaufzeit).

## 2. Gaming, Allround & Office-Arbeit
- **Empfehlung: Windows 11 + WSL2 (Windows Subsystem for Linux)**:
  - Warum: Volle Kompatibilität mit allen Anti-Cheat-Spielen und Windows-Apps + vollwertiges Linux-Terminal im Hyper-V mit direkter GPU-Passthrough-Unterstützung für Entwicklung.

## 3. Datenschutz, Privatsphäre & Sicherheit
- **Desktop: Linux (Fedora mit Vollverschlüsselung LUKS oder Tails/Qubes OS)**:
  - Keine Telemetrie, Open-Source Auditierbarkeit, Sandboxing via Flatpak.
- **Mobile: Android mit GrapheneOS (auf Google Pixel Hardware)**:
  - Vollständige De-Google-Option, gehärteter Speicherschutz (Hardened Malloc), granular entziehbare Netzwerk- und Sensor-Berechtigungen pro App.

## 4. Kreative Workflows (Audio, Video, Design) & Mobile Synergie
- **Desktop: macOS (MacBook Pro / Mac Studio)**:
  - Nahtlose Farbkalibrierung (Display P3), CoreAudio-Engine mit unschlagbaren Latenzen, professionelle Suite (Final Cut, Logic Pro, Adobe).
- **Mobile: iOS / iPadOS**:
  - Nahtloses Ökosystem (AirDrop, Universal Clipboard, Continuity), lange Update-Garantie, intuitive Bedienung.""",

    # 3. Mobile OS Internals: Android vs iOS
    r"""# Mobile Betriebssystem-Architektur: Android (AOSP) vs iOS (Darwin/Cocoa Touch):

## 1. Android Internals (AOSP & Linux Kernel)
- **Architektur-Schichten**:
  1. Linux Kernel mit Android-spezifischen Treibern (Binder IPC, Ashmem).
  2. Hardware Abstraction Layer (HAL).
  3. Android Runtime (ART) mit Ahead-of-Time (AOT) und JIT Profiling.
  4. Zygote-Prozess: Lädt System-Ressourcen vorab und forkt neue App-Prozesse blitzschnell.
- **Entwickler- & Power-User-Möglichkeiten**:
  - `Termux`: Vollwertige Linux-CLI auf dem Smartphone (Python, Git, Node.js, C-Compiler).
  - Rooting via Magisk / KernelSU für Kernel-Level Modifikationen.

## 2. iOS Internals (Darwin / Secure Enclave)
- **App Sandboxing**: Jede App läuft in einer isolierten Sandbox ohne Zugriff auf das Dateisystem anderer Apps (außer über System-Picker).
- **Secure Enclave Processor (SEP)**: Dedizierter Hardware-Sicherheitschip für biometrische Daten (Face ID / Touch ID) und kryptographische Schlüssel.
- **Jetsam Memory Killer**: Striktes Speichermanagement, das ressourcenhungrige Hintergrundprozesse sofort terminiert, um 60/120 FPS UI-Flüssigkeit zu garantieren."""
]


def run_operating_systems_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/cyber_web_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 300,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("💻 MASSIVE OPERATING SYSTEMS & PLATFORMS PIPELINE (7B PRE-TRAINING)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende Cyber/OS-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

    vocab = MultiGranularVocabulary.load_json(vocab_file)
    tokenizer = ViterbiTokenizer(vocab)
    encoder = BitstreamEncoder(vocab_size=vocab.size, bit_width=16)

    buffer_tokens: List[int] = []
    total_tokens_written = shard_count * max_tokens_per_shard
    start_time = time.time()

    def flush_shard():
        nonlocal shard_count, buffer_tokens, total_tokens_written
        if not buffer_tokens:
            return
        shard_path = os.path.join(output_dir, f"cyber_web_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [OS/SYS/PLATFORM Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Deep OS Architecture & Decision Standards
    print("\n📚 [Quelle 1/2] Tokenisiere Linux Distros, macOS Apple Silicon, Windows WSL2 & Mobile OS...", flush=True)
    for doc in OS_COMPREHENSIVE_KNOWLEDGE:
        formatted = f"### Betriebssystem-Architektur & Plattform-Entscheidungsmatrix:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest OS, Shell, Sysadmin & Platform Instructions
    print("\n🐧 [Quelle 2/2] Streame Linux Shell, Kernel, Sysadmin & Multi-Platform Dialoge...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        os_count = 0
        for item in code_ds:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            if not instr or not resp:
                continue

            keywords = ["linux", "ubuntu", "arch", "fedora", "debian", "macos", "windows", "wsl", "android", "ios", "kernel", "bash", "systemd", "zfs", "btrfs", "driver", "package"]
            is_relevant = any(kw in instr.lower() or kw in resp.lower() for kw in keywords)
            if not is_relevant:
                continue

            formatted = f"### Benutzer (Operating Systems & System-Wahl):\n{instr}\n\n### Assistent (Principal OS Architect & Systems Engineer):\n<think>\nAnalysiere Betriebssystem-Architektur, Hardware-Kompatibilität, Performance und Einsatzzweck:\n</think>\n{resp}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            os_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if os_count % 2000 == 0:
                print(f"  ⚙️ [OS-Engine] {os_count:,} Betriebssystem-Dialoge verarbeitet (Shards: {shard_count})", flush=True)

        print(f"✅ OS & Platform Dialoge abgeschlossen ({os_count:,} verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei OS Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Operating Systems Specialist Ingestion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_operating_systems_ingestion()
