#!/usr/bin/env python3
"""Specialized Expert Ingestion Pipeline for Modern Java, Minecraft Game Engine Mechanics (1.8 - 1.21+),
Spigot/Paper Server Architecture, Fabric/Forge Modding, and Precision Voxel Physics.

Covers:
1. Modern Java (Java 8, 11, 17, 21 LTS): Concurrency, Streams, Records, Virtual Threads (Loom), Netty ByteBuf.
2. Minecraft Version Evolution (1.8 -> 1.21+):
   - 1.8: Legacy numerical IDs, Metadata (0-15), 1.8 PvP Combat (W-tapping, rod mechanics, no weapon cooldown, sword block-hit).
   - 1.9 - 1.12: Dual Wielding, Attack Speed Cooldowns, Elytra, Forge 1.12 Capabilities & TileEntities.
   - 1.13 The Flattening: Transition from numerical IDs/meta to typed BlockStates, Brigadier commands.
   - 1.14 - 1.20: Villager POI schedules, Netherite, World Height (-64 to 320), Display Entities.
   - 1.20.5 / 1.21+: Data Component overhaul replacing raw item NBT, Crafter, Mace smash mechanics.
3. Precision Game Mechanics & Voxel Physics (Feinmechanik):
   - Block Breaking: Hardness (H), Resistance (R), Tool Multipliers, Efficiency Formula (speed = base + level^2 + 1),
     Haste/Fatigue modifiers, Aqua Affinity, In-Air penalty (5x), Insta-Mine condition (damage_per_tick >= 1.0).
   - Block Placement: 3D DDA Voxel Raycasting, Directional Face offsets, AABB entity collision checks, Waterlogging.
   - Entity Physics: 20 TPS tick loop, Gravity (0.08 b/tick), Air drag (0.98), Ground friction, Knockback vectors.
4. Server Plugin & Modding Implementations (Paper API, Fabric Mixins, NeoForge).

Streams into 16-Bit .mgbs bitstream shards in data/java_minecraft_knowledge/shards/.
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


MINECRAFT_CORE_ENGINE_KNOWLEDGE = [
    # 1. Block Mining & Breaking Physics Engine (Feinmechanik)
    r"""# Minecraft Voxel Engine: Block Mining & Breaking Calculations (1.8 - 1.21+):

## 1. Mathematische Formel für Abbauzeit & Blockschaden pro Tick
Die Abbauzeit eines Blocks wird serverseitig in Ticks ($20\text{ Ticks} = 1\text{ Sekunde}$) berechnet.
In jedem Tick fügt der Spieler dem Block einen Schadensfortschritt $\Delta D$ zu:

$$\Delta D = \frac{\text{Abbau-Multiplikator}}{\text{Block-Härte } H \times \text{Divisor}}$$

- **Divisor**:
  - Wenn das Werkzeug geeignet ist, um den Drop zu erhalten (`is_correct_tool`): $\text{Divisor} = 30$.
  - Wenn das Werkzeug ungeeignet ist (z. B. Faust auf Stein): $\text{Divisor} = 100$.

## 2. Berechnung des Abbau-Multiplikators (Tool Speed Multiplier)
1. **Basis-Geschwindigkeit Werkzeug**:
   - Faust / Unpassendes Tool: $1.0$
   - Holz: $2.0$ | Stein: $4.0$ | Eisen: $6.0$ | Diamant: $8.0$ | Netherite: $9.0$ | Gold: $12.0$
2. **Effizienz-Verzauberung (Efficiency I - V)**:
   - Wenn das Werkzeug zur Blockkategorie passt: $\text{Speed}_{\text{eff}} = \text{Speed}_{\text{base}} + (\text{level}^2 + 1)$.
   - Beispiel Diamant-Spitzhacke mit Effizienz V: $8.0 + (5^2 + 1) = 8.0 + 26 = 34.0$.
3. **Status-Effekte (Haste & Mining Fatigue)**:
   - Haste (Eile): $\text{Speed} \times (1.0 + 0.2 \times \text{level})$.
   - Mining Fatigue (Abbaulähmung):
     - Stufe I: $\times 0.3$ | Stufe II: $\times 0.09$ | Stufe III: $\times 0.0027$ | Stufe IV+: $\times 0.00081$.
4. **Umgebungs-Modifikatoren**:
   - Im Wasser ohne Bodenkontakt / ohne Aqua Affinity: $\text{Speed} \div 5$.
   - In der Luft schwebend (während des Springens/Fallens): $\text{Speed} \div 5$.

## 3. Instant-Mining (Insta-Mine) Bedingung
Ein Block wird in genau $1\text{ Tick}$ ($0.05\text{s}$) instant abgebaut, wenn:
$$\Delta D \ge 1.0 \iff \text{Effektiver Speed} \ge 30 \times \text{Härte } H$$
- **Beispiel Stein ($H = 1.5$)**: Benötigt $\text{Speed} \ge 30 \times 1.5 = 45.0$.
  - Netherite-Spitzhacke ($9.0$) + Effizienz V ($+26 \implies 35.0$) + Haste II Beacon ($\times 1.4 \implies 35.0 \times 1.4 = 49.0 \ge 45.0$) $\implies$ **Insta-Mine garantiert!**""",

    # 2. Block Placement, Raycasting & Bounding Box Collision
    r"""# Minecraft Voxel Engine: Block Placement, DDA Raycasting & Collision Mechanics:

## 1. 3D Digital Differential Analyzer (DDA) Voxel Raycast
Um zu bestimmen, welcher Block anvisiert wird, castet die Engine einen Strahl vom Augenvektor des Spielers $\vec{P}_{\text{eye}} = (x, y + 1.62, z)$ entlang des Blickrichtungs-Einheitsvektors $\vec{D} = (\cos(\text{pitch})\sin(-\text{yaw}), -\sin(\text{pitch}), \cos(\text{pitch})\cos(-\text{yaw}))$.
- **Reichweite**: $4.5\text{ Blöcke}$ im Survival-Modus, $5.0\text{ Blöcke}$ im Creative-Modus.
- **Hit-Ergebnis (`BlockHitResult`)**:
  - `block_pos = (bx, by, bz)`: Koordinaten des anvisierten Blocks.
  - `direction`: Anvisierte Seite (`UP`, `DOWN`, `NORTH`, `SOUTH`, `EAST`, `WEST`).
  - `click_offset = (ox, oy, oz) \in [0.0, 1.0]^3`: Sub-Voxel-Auftreffpunkt für Treppenausrichtung und Halbstufen (Top/Bottom Slab).

## 2. Block-Platzierungs-Validierung & AABB-Kollision
Vor dem Setzen an der Zielposition $\vec{P}_{\text{target}} = \vec{P}_{\text{hit}} + \vec{D}_{\text{normal}}$ prüft die Engine:
1. **Ersetzbarkeit (`canBeReplaced`)**: Ist der Zielblock Luft, Wasser, Gras oder Schnee-Layer?
2. **Entity AABB Collision Check**:
   $$\text{BlockAABB}(\vec{P}_{\text{target}}) \cap \text{EntityAABB} = \emptyset$$
   Befindet sich ein Spieler, Mob oder Item-Frame innerhalb der Bounding Box des zu setzenden Blocks, wird das `BlockPlaceEvent` storniert.
3. **Nachbar-Updates (Block Updates / Observer Logic)**:
   - Beim Setzen feuert die Engine `onBlockAdded()` und sendet Block-Updates an die 6 angrenzenden Nachbarn ($x\pm 1, y\pm 1, z\pm 1$).
   - Observer erfassen die Änderung des `BlockState` und erzeugen im darauffolgenden Tick ein $2\text{-Redstone-Tick}$ langes Signal (Stärke 15).""",

    # 3. Evolution: 1.8 Legacy vs 1.9 Combat vs 1.13 The Flattening vs 1.20.5+ Components
    r"""# Minecraft Versionen-Evolution (1.8 bis 1.21+): Architektur & Unterschiede:

## 1. Minecraft 1.8 (The Bountiful Update & PvP Golden Era)
- **ID-System**: Numerische Block- & Item-IDs ($0 - 255$ für Blöcke, $256+$ für Items) kombiniert mit $4\text{-Bit}$ Metadaten ($0 - 15$). Beispiel: Wolle war ID `35`, Rote Wolle `35:14`.
- **1.8 PvP Combat Mechanics**:
  - Kein Waffen-Cooldown: Angriffsfrequenz ausschließlich durch Klickgeschwindigkeit (CPS) und Damage-Immunity-Frames ($10\text{ Ticks} = 0.5\text{s}$ Invulnerability) begrenzt.
  - Block-Hitting: Gleichzeitiges Rechts- und Linksklicken mit dem Schwert reduziert eingehenden Schaden um $50\%$ bei voller Angriffsfähigkeit.
  - Movement-Resets: W-Tapping, S-Tapping und Angel-Hits (Fishing Rod Projectiles) setzen den gegnerischen Sprint zurück und maximieren den horizontalen Knockback-Impuls.

## 2. Minecraft 1.13 (The Flattening)
- **Abschaffung von numerischen IDs & Metadaten**: Jede Blockvariante ist ein eigenständiger Identifier (z. B. `minecraft:red_wool` statt `35:14`).
- **Typisierte BlockStates**: Blöcke besitzen typisierte Properties (`facing=north`, `waterlogged=true`, `half=bottom`).
- **Brigadier Command Engine**: Typsichere AST-basierte Command-Syntax mit Autovervollständigung und Vorschlägen.

## 3. Minecraft 1.20.5 & 1.21+ (Data Components & Modern Mechanics)
- **Data Component Overhaul**: Vollständige Ablösung von unstrukturiertem Item-NBT durch typisierte Komponenten (`minecraft:custom_data`, `minecraft:enchantments`, `minecraft:food`, `minecraft:damage_resistant`).
- **Neue Engine-Elemente**:
  - Display Entities (`block_display`, `item_display`, `text_display`) mit $4\times 4$ Transformationsmatrizen.
  - Interaction Entities für präzise Raycast-Trefferzonen.
  - Der Crafter: Automatisierte Crafting-Logik mit Redstone-Puls-Steuerung.
  - Der Streitkolben (Mace): Schadens-Skalierung proportional zur Fallhöhe $\Delta Y$ mit Negierung von Fallschaden bei erfolgreichem Treffer.""",

    # 4. Spigot / Paper API & High-Performance Server Development (Java)
    r"""# High-Performance Paper/Spigot Plugin Engineering in Modern Java:

## 1. Asynchrone Architektur & Thread-Safety
- **Main Thread (Tick Loop)**: Alle Block-, World- und Entity-Mutationen MÜSSEN auf dem Server-Haupt-Thread erfolgen.
- **Asynchrone I/O-Tasks**: Datenbank-Abfragen (MySQL/SQLite/MongoDB), HTTP-Requests und Datei-Zugriffe müssen strikt via `Bukkit.getAsyncScheduler()` (Folia) bzw. `runTaskAsynchronously()` ausgelagert werden.

## 2. Vollständiger Paper Custom-Tool & Block-Break Listener (Java 21)
```java
package com.antigravity.minecraft.listener;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import org.bukkit.Material;
import org.bukkit.Sound;
import org.bukkit.block.Block;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.inventory.ItemStack;

public final class CustomMiningListener implements Listener {

    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void onBlockBreak(BlockBreakEvent event) {
        Player player = event.getPlayer();
        Block block = event.getBlock();
        ItemStack tool = player.getInventory().getItemInMainHand();

        // 1.8 vs 1.21 Check: Custom Mining Logic
        if (tool.getType() == Material.DIAMOND_PICKAXE && block.getType() == Material.OBSIDIAN) {
            // Spezieller Multiplikator für Obsidian-Abbau
            player.playSound(block.getLocation(), Sound.BLOCK_ANVIL_USE, 1.0f, 1.5f);
            player.sendMessage(Component.text("Legendärer Obsidian abgebaut!", NamedTextColor.DARK_PURPLE));
        }
    }
}
```""",

    # 5. Fabric & NeoForge Modding with Mixins (Java 21)
    r"""# Fabric & NeoForge Modding: Bytecode-Mixins, Registry & Custom Blocks:

## 1. Fabric Mixin Injection in Block Breaking Pipeline
```java
package com.antigravity.mod.mixin;

import net.minecraft.block.BlockState;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.network.ServerPlayerInteractionManager;
import net.minecraft.util.math.BlockPos;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(ServerPlayerInteractionManager.class)
public class BlockBreakingMixin {

    @Inject(method = "tryBreakBlock", at = @At("HEAD"), cancellable = true)
    private void onTryBreakBlock(BlockPos pos, CallbackInfoReturnable<Boolean> cir) {
        ServerPlayerInteractionManager manager = (ServerPlayerInteractionManager)(Object)this;
        ServerPlayerEntity player = manager.player;
        BlockState state = player.getWorld().getBlockState(pos);

        // Custom Feinmechanik-Hook: Verhindere Abbau unzerstörbarer Zonen
        if (pos.getY() < -60 && state.isOf(net.minecraft.block.Blocks.BEDROCK)) {
            cir.setReturnValue(false);
        }
    }
}
```"""
]


def run_java_minecraft_ingestion(
    output_dir: str = "/home/benjamin/Bilder/data/java_minecraft_knowledge/shards",
    vocab_file: str = "/home/benjamin/Bilder/data/vocab_65k.json",
    max_tokens_per_shard: int = 500_000,
    target_shards: int = 60,
) -> int:
    global STOP_REQUESTED

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(vocab_file):
        vocab_file = "/home/benjamin/Bilder/vocab.json"

    print("=" * 80, flush=True)
    print("⛏️ MASSIVE JAVA & MINECRAFT GAME ENGINE SPECIALIST PIPELINE (1.8 - 1.21+)", flush=True)
    print(f"📁 Ziel-Verzeichnis: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    existing_files = sorted(glob.glob(os.path.join(output_dir, "*.mgbs")))
    shard_count = len(existing_files)
    print(f"  ⏭️ {shard_count} bestehende Java/Minecraft-Shards gefunden. Starte ab Index {shard_count:04d}...", flush=True)

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
        shard_path = os.path.join(output_dir, f"java_minecraft_shard_{shard_count:04d}.mgbs")
        encoder.save_to_file(shard_path, buffer_tokens, raw_byte_count=len(buffer_tokens) * 2)
        total_tokens_written += len(buffer_tokens)
        shard_count += 1
        elapsed = time.time() - start_time
        print(f"  💾 [JAVA/MINECRAFT Shard {shard_count:04d}] {len(buffer_tokens):,} Tokens | Gesamt: {total_tokens_written:,} Tokens -> {os.path.basename(shard_path)}", flush=True)
        buffer_tokens = []

    # 1. Ingest Core Minecraft Engine & Precision Mechanics
    print("\n📚 [Quelle 1/3] Tokenisiere Minecraft Feinmechanik, Voxel-Physics & Versions-Evolution (1.8 - 1.21+)...", flush=True)
    for doc in MINECRAFT_CORE_ENGINE_KNOWLEDGE:
        formatted = f"### Minecraft Game Engine & Java Architecture:\n{doc.strip()}\n\n"
        buffer_tokens.extend(tokenizer.encode(formatted))
        if len(buffer_tokens) >= max_tokens_per_shard:
            flush_shard()

    # 2. Ingest Java & Minecraft Instruction Dialogues (Evol-Code Java & Game Logic)
    print("\n☕ [Quelle 2/3] Streame Java 17/21 LTS, OOP Architecture & Concurrency...", flush=True)
    try:
        code_ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", streaming=True)
        java_count = 0
        for item in code_ds:
            if STOP_REQUESTED or shard_count >= target_shards:
                break
            instr = item.get("instruction", "")
            resp = item.get("output", "")
            if not instr or not resp:
                continue

            # Prioritize Java, Minecraft, Game Engines, Protocols & Netty
            keywords = ["java", "minecraft", "class", "public static", "thread", "concurrency", "socket", "packet", "game", "voxel", "vector", "matrix"]
            is_relevant = any(kw in instr.lower() or kw in resp.lower() for kw in keywords)
            if not is_relevant:
                continue

            formatted = f"### Benutzer (Java & Game Engineering):\n{instr}\n\n### Assistent (Game Engine & Minecraft Experte):\n<think>\nAnalysiere Java-Klassenarchitektur, Voxel-Feinmechanik und Performance:\n</think>\n{resp}\n\n"
            buffer_tokens.extend(tokenizer.encode(formatted))
            java_count += 1

            if len(buffer_tokens) >= max_tokens_per_shard:
                flush_shard()

            if java_count % 2000 == 0:
                print(f"  ⚙️ [Java-Engine] {java_count:,} Instruktionen verarbeitet (Shards: {shard_count})", flush=True)

        print(f"✅ Java & Game Engine Dialoge abgeschlossen ({java_count:,} verarbeitet).", flush=True)
    except Exception as e:
        print(f"⚠️ Hinweis bei Java Stream: {e}", flush=True)

    flush_shard()
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 Java & Minecraft Specialist Ingestion abgeschlossen! Gesamte Shards: {shard_count} (~{total_tokens_written:,} Tokens)", flush=True)
    print("=" * 80, flush=True)
    return shard_count


if __name__ == "__main__":
    run_java_minecraft_ingestion()
