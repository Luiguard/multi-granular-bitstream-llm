#!/usr/bin/env python3
"""
Constructs the Canonical 20-Bit Multi-Granular Vocabulary (1,048,576 Tokens):
- Tier 0 (0-255): 256 Raw Byte Fallbacks (100% Zero-OOV guarantee).
- Tier 1: Core Multilingual Subwords & BPE (NLLB-200 for 204 languages + OpenAI cl100k_base for Code/STEM).
- Tier 2: Real Natural Dictionary Words from 60+ World Languages (HermitDave OpenSubtitles Frequency Dictionaries)
          + German Compound Words & International Scientific Terminology.
- Tier 3: Code Templates, Multi-space Indents, Reasoning Tags (<think>, </think>), Minecraft Engine AST Patterns
          and Reserved Expansion Slots.

Outputs:
1. data/vocab_1m_20bit.json (Canonical JSON Golden Master)
2. data/vocab_1m_20bit.bin (Ultra-fast binary format for sub-50ms loading)
"""

import os
import sys
import json
import time
import struct
import hashlib
import glob
from typing import Dict, List, Set, Tuple

sys.path.insert(0, "/home/benjamin/Bilder")
from pipeline.vocabulary import MultiGranularVocabulary, TokenTier

TARGET_VOCAB_SIZE = 1048576  # 2^20
RAW_DIR = "/home/benjamin/Bilder/data/dictionaries_raw"
OUTPUT_JSON = "/home/benjamin/Bilder/data/vocab_1m_20bit.json"
OUTPUT_BIN = "/home/benjamin/Bilder/data/vocab_1m_20bit.bin"


def save_binary_vocab(vocab: MultiGranularVocabulary, filepath: str) -> str:
    """Serializes the vocabulary into an ultra-fast compact binary format.
    Format:
    - Magic: b'MG20' (4 bytes)
    - Version: uint16 (1)
    - Vocab Size: uint32 (1048576)
    - Entries: [tier: uint8, byte_len: uint16, freq: uint32, pmi: float32, text_bytes: utf-8]
    """
    hasher = hashlib.sha256()
    with open(filepath, "wb") as f:
        header = struct.pack("<4sHI", b"MG20", 1, vocab.size)
        f.write(header)
        hasher.update(header)

        for token_id in range(vocab.size):
            text = vocab.id_to_token[token_id]
            text_bytes = text.encode("utf-8")
            tier = int(vocab.id_to_tier[token_id])
            byte_len = vocab.id_to_byte_len[token_id]
            freq = vocab.id_to_frequency.get(token_id, 1)
            pmi = float(vocab.id_to_pmi.get(token_id, 0.0))

            entry_hdr = struct.pack("<BHIffH", tier, byte_len, freq, pmi, 0.0, len(text_bytes))
            f.write(entry_hdr)
            f.write(text_bytes)
            hasher.update(entry_hdr)
            hasher.update(text_bytes)

    return hasher.hexdigest()


def build_1m_20bit_vocabulary():
    start_time = time.time()
    print("=" * 85)
    print("🌍 ERSTELLE KANONISCHES 20-BIT MULTI-GRANULAR VOKABULAR (1.048.576 TOKENS)")
    print("=" * 85)

    vocab = MultiGranularVocabulary()
    seen_tokens: Set[str] = set()

    # -------------------------------------------------------------------------
    # 1. Tier 0: 256 Exact Raw Bytes (0x00 to 0xFF)
    # -------------------------------------------------------------------------
    for b in range(256):
        byte_char = bytes([b]).decode("latin1")
        seen_tokens.add(byte_char)

    print(f"  • Tier 0 Raw Bytes registriert: {vocab.size} Tokens (100% Zero-OOV)")

    # -------------------------------------------------------------------------
    # 2. Tier 3: Core Cognitive Tags, Templates & Syntactic Indentations
    # -------------------------------------------------------------------------
    special_templates = [
        "<think>", "</think>", "<reasoning>", "</reasoning>",
        "<reward_plus>", "<reward_minus>", "<sandbox_exec>", "</sandbox_exec>",
        "<vision_patch>", "</vision_patch>", "<bitstream_child>", "</bitstream_child>",
        "  ", "    ", "      ", "        ", "          ", "            ", "                ",
        "\n  ", "\n    ", "\n      ", "\n        ", "\n            ", "\n                ",
        "def __init__(self, ", "def __call__(self, ", "def forward(self, ",
        "if __name__ == '__main__':\n", "public static void main(String[] args) {\n",
        "import torch\nimport torch.nn as nn\n", "import torch.nn.functional as F\n",
        "from typing import Dict, List, Optional, Tuple, Any, Union\n",
        "export default function ", "async function ", "const [state, setState] = useState(",
        "SELECT * FROM ", "ORDER BY ", "GROUP BY ", "WHERE id = ",
        "\\begin{equation}\n", "\\end{equation}\n", "\\frac{", "}^{2}",
        "https://de.wikipedia.org/wiki/", "https://en.wikipedia.org/wiki/",
        "G_{μν} + Λ g_{μν} = \\frac{8π G}{c^4} T_{μν}",
    ]

    code_keywords = [
        "import ", "from ", "def ", "class ", "return ", "yield ", "async ", "await ",
        "try:\n    ", "except Exception as e:\n    ", "finally:\n    ", "with open(",
        "fn ", "pub fn ", "let mut ", "impl ", "struct ", "enum ", "match ", "Ok(", "Err(",
        "public class ", "private final ", "protected ", "throw new ", "catch (Exception ",
        "template <typename T>", "std::vector<", "std::shared_ptr<", "std::unique_ptr<",
        "<!DOCTYPE html>", "<html lang=\"de\">", "<div class=\"", "<span class=\"",
    ]

    for tmpl in special_templates + code_keywords:
        if tmpl not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE:
            vocab.add_token(tmpl, tier=TokenTier.TEMPLATE, frequency=100000, pmi=10.0)
            seen_tokens.add(tmpl)

    print(f"  • Tier 3 Spezial-Templates registriert: {vocab.size:,} Tokens")

    # -------------------------------------------------------------------------
    # 3. Tier 1: Multilingual Subwords from Meta NLLB-200 (204 ISO Languages)
    # -------------------------------------------------------------------------
    nllb_file = os.path.join(RAW_DIR, "nllb_tokens.json")
    if os.path.exists(nllb_file):
        with open(nllb_file, "r", encoding="utf-8") as f:
            nllb_tokens = json.load(f)
        added_nllb = 0
        for t in nllb_tokens:
            if t and t not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE:
                tier = TokenTier.WORD if len(t.split()) <= 1 else TokenTier.PHRASE
                vocab.add_token(t, tier=tier, frequency=50000, pmi=4.0)
                seen_tokens.add(t)
                added_nllb += 1
        print(f"  • Tier 1 NLLB-200 Multilingual Subwords hinzugefügt: {added_nllb:,} Tokens (Stand: {vocab.size:,})")

    # -------------------------------------------------------------------------
    # 4. Tier 1: Core Subwords & Code Tokens from cl100k_base
    # -------------------------------------------------------------------------
    try:
        import tiktoken
        cl100k = tiktoken.get_encoding("cl100k_base")
        added_bpe = 0
        for token_id in range(cl100k.n_vocab):
            if vocab.size >= TARGET_VOCAB_SIZE:
                break
            try:
                b_str = cl100k.decode_bytes([token_id])
                s_str = b_str.decode("utf-8")
                if s_str and s_str not in seen_tokens:
                    tier = TokenTier.WORD if len(s_str.split()) <= 1 else TokenTier.PHRASE
                    vocab.add_token(s_str, tier=tier, frequency=40000, pmi=4.0)
                    seen_tokens.add(s_str)
                    added_bpe += 1
            except Exception:
                continue
        print(f"  • Tier 1 cl100k BPE & STEM Subwords hinzugefügt: {added_bpe:,} Tokens (Stand: {vocab.size:,})")
    except Exception as e:
        print(f"  ⚠️ Hinweis cl100k: {e}")

    # -------------------------------------------------------------------------
    # 5. Tier 2: Real Natural Dictionary Words from HermitDave (60+ Languages)
    # -------------------------------------------------------------------------
    word_files = sorted(glob.glob(os.path.join(RAW_DIR, "*_words.json")))
    print(f"  • Lade Frequenz-Wörterbücher aus {len(word_files)} Sprachdateien...", flush=True)

    # Load all word lists
    lang_word_queues: Dict[str, List[Tuple[str, int]]] = {}
    for wf in word_files:
        lang_code = os.path.basename(wf).replace("_words.json", "")
        try:
            with open(wf, "r", encoding="utf-8") as f:
                items = json.load(f)
                lang_word_queues[lang_code] = [(it["word"], it.get("freq", 1)) for it in items]
        except Exception:
            continue

    # Fair round-robin insertion across all languages
    max_depth = max(len(q) for q in lang_word_queues.values()) if lang_word_queues else 0
    added_dict_words = 0
    for depth in range(max_depth):
        if vocab.size >= TARGET_VOCAB_SIZE - 20000:
            break
        for lang, words in lang_word_queues.items():
            if depth < len(words):
                w, freq = words[depth]
                if w and len(w) > 1 and w not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE - 20000:
                    vocab.add_token(w, tier=TokenTier.WORD, frequency=freq, pmi=3.0)
                    seen_tokens.add(w)
                    added_dict_words += 1
                # Also add leading space version for natural language fluency
                w_space = " " + w
                if w_space not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE - 20000:
                    vocab.add_token(w_space, tier=TokenTier.WORD, frequency=freq, pmi=3.0)
                    seen_tokens.add(w_space)
                    added_dict_words += 1

    print(f"  • Tier 2 Wörterbuch-Lemmata aus 60+ Sprachen hinzugefügt: {added_dict_words:,} Tokens (Stand: {vocab.size:,})")

    # -------------------------------------------------------------------------
    # 6. Tier 2: German Scientific Lexemes & Compound Words
    # -------------------------------------------------------------------------
    compound_stems = [
        "Quanten", "Relativitäts", "Wellen", "Wahrscheinlichkeits", "Entropie",
        "Differential", "Integral", "Vektor", "Matrix", "Tensor", "Hilbert",
        "Konvergenz", "Gradienten", "Optimierungs", "Verfassungs", "Sicherheits",
        "Überwachungs", "Entscheidungs", "Verwaltungs", "Transformations", "Architektur",
        "Multiprozessor", "Mikroarchitektur", "Echtzeit", "Netzwerk", "Protokoll",
        "Thermodynamik", "Elektrodynamik", "Magnetohydrodynamik", "Astrophysik",
        "Molekular", "Genom", "Neuronale", "Kognitions", "Synapsen", "Topologie"
    ]
    compound_suffixes = [
        "theorie", "mechanik", "funktion", "verteilung", "gleichung", "raum",
        "tensor", "operator", "kalkül", "algorithmus", "struktur", "muster",
        "gesetz", "urteil", "behörde", "bericht", "analyse", "prozess", "schicht",
        "zustand", "dynamik", "invarianz", "transformation", "potenzial"
    ]
    added_compounds = 0
    for stem in compound_stems:
        for suf in compound_suffixes:
            for pref in ["", " "]:
                comp = f"{pref}{stem}{suf}"
                if comp not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE - 10000:
                    vocab.add_token(comp, tier=TokenTier.WORD, frequency=500, pmi=4.0)
                    seen_tokens.add(comp)
                    added_compounds += 1

    print(f"  • Tier 2 Deutsche Wissenschafts-Komposita hinzugefügt: {added_compounds:,} Tokens (Stand: {vocab.size:,})")

    # -------------------------------------------------------------------------
    # 7. Tier 3: Minecraft Game Engine, Spigot API & Voxel Physics
    # -------------------------------------------------------------------------
    minecraft_tokens = [
        "minecraft:", "BlockState", "BlockBreakEvent", "ItemStack", "PlayerInteractEvent",
        "Material.DIAMOND_PICKAXE", "Material.OBSIDIAN", "Material.NETHERITE_SWORD",
        "DamageModifier", "AquaAffinity", "EfficiencyMultiplier", "InstaMineCondition",
        "EntityAABB", "RayTraceResult", "VoxelShape", "WorldHeightLimit", "DataComponent",
        "tick_loop", "damage_per_tick", "tool_multiplier", "block_hardness", "block_resistance"
    ]
    for m_tok in minecraft_tokens:
        for pref in ["", " "]:
            tok = f"{pref}{m_tok}"
            if tok not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE:
                vocab.add_token(tok, tier=TokenTier.PHRASE, frequency=1000, pmi=5.0)
                seen_tokens.add(tok)

    # -------------------------------------------------------------------------
    # 7b. Comprehensive Scientific Symbols: Math, Physics, Chemistry & LaTeX
    # -------------------------------------------------------------------------
    math_symbols = [
        "≠", "≤", "≥", "≦", "≧", "≪", "≫", "±", "∓", "×", "÷", "·", "∘", "†", "‡",
        "∀", "∃", "∄", "∈", "∉", "∋", "∌", "∅", "⊂", "⊃", "⊆", "⊇", "∪", "∩", "∧", "∨", "¬", "⊕", "⊖", "⊗", "⊘", "⊙",
        "∂", "∇", "∫", "∬", "∭", "∮", "∯", "∰", "∑", "∏", "∐", "√", "∛", "∜", "∞", "∝", "∠", "∟", "⊥", "∥",
        "∼", "∽", "≃", "≅", "≈", "≉", "≍", "≎", "≏", "≐", "≒", "≓", "≔", "≕", "≡", "≢",
        "→", "←", "↔", "⇒", "⇐", "⇔", "↦", "↤", "↑", "↓", "↗", "↘", "↖", "↙", "⇄", "⇆", "⇋", "⇌",
        "ℝ", "ℂ", "ℕ", "ℤ", "ℚ", "ℙ", "ℍ", "𝔽",
        "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉", "₊", "₋", "₌", "₍", "₎", "ₐ", "ₑ", "ₒ", "ₓ", "ₔ", "ᵢ", "ᵣ", "ᵤ", "ᵥ", "ⱼ", "ₖ", "ₗ", "ₘ", "ₙ", "ₚ", "ₛ", "ₜ",
        "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹", "⁺", "⁻", "⁼", "⁽", "⁾", "ⁿ", "ⁱ", "ʲ", "ᵏ"
    ]

    physics_chem = [
        "ℏ", "Å", "℃", "℉", "Ω", "µ", "Å",
        "eV", "keV", "MeV", "GeV", "TeV", "mol", "kJ/mol", "J/mol",
        "m/s", "m/s²", "kg/m³", "N/m", "J/K", "W/m²", "Pa", "kPa", "MPa", "GPa", "bar", "mbar", "atm",
        "Hz", "kHz", "MHz", "GHz", "THz", "ppm", "ppb",
        "α-decay", "β-minus", "β-plus", "γ-radiation", "e⁻", "e⁺", "p⁺", "n⁰", "ν_e", "ν_μ", "ν_τ", "W⁺", "W⁻", "Z⁰", "H⁰",
        "⇌", "⇄", "⇆", "⇋", "⟶", "⟵", "⟷", "∆", "ΔH", "ΔS", "ΔG", "pH", "pOH", "pKa", "pKb", "K_w", "K_sp",
        "(s)", "(l)", "(g)", "(aq)",
        "H₂O", "CO₂", "H₂SO₄", "HNO₃", "HCl", "H₃PO₄", "NaOH", "KOH", "NH₃", "NH₄⁺",
        "CH₄", "C₂H₆", "C₂H₄", "C₂H₂", "C₆H₆", "C₆H₁₂O₆", "NaCl", "KCl", "CaCl₂", "CaCO₃",
        "Fe²⁺", "Fe³⁺", "Cu²⁺", "Ag⁺", "SO₄²⁻", "NO₃⁻", "CO₃²⁻", "PO₄³⁻", "OH⁻", "H₃O⁺", "CN⁻"
    ]

    latex_commands = [
        "\\frac{", "\\sqrt{", "\\sum_{", "\\prod_{", "\\int_{", "\\iint_{", "\\iiint_{", "\\oint_{", "\\partial ", "\\nabla ", "\\cdot ",
        "\\alpha ", "\\beta ", "\\gamma ", "\\delta ", "\\epsilon ", "\\zeta ", "\\eta ", "\\theta ", "\\iota ", "\\kappa ", "\\lambda ", "\\mu ", "\\nu ", "\\xi ", "\\pi ", "\\rho ", "\\sigma ", "\\tau ", "\\upsilon ", "\\phi ", "\\chi ", "\\psi ", "\\omega ",
        "\\Gamma ", "\\Delta ", "\\Theta ", "\\Lambda ", "\\Xi ", "\\Pi ", "\\Sigma ", "\\Upsilon ", "\\Phi ", "\\Psi ", "\\Omega ",
        "\\hbar ", "\\infty ", "\\approx ", "\\neq ", "\\leq ", "\\geq ", "\\pm ", "\\mp ", "\\times ", "\\div ", "\\equiv ", "\\sim ", "\\propto ", "\\in ", "\\notin ", "\\subset ", "\\subseteq ", "\\cup ", "\\cap ", "\\forall ", "\\exists ",
        "\\vec{", "\\hat{", "\\dot{", "\\ddot{", "\\bar{", "\\tilde{", "\\mathbf{", "\\mathcal{", "\\mathbb{", "\\text{",
        "\\rightleftharpoons ", "\\xrightarrow{", "\\xleftarrow{", "\\ce{",
        "\\begin{equation}", "\\end{equation}", "\\begin{align}", "\\end{align}",
        "\\begin{matrix}", "\\end{matrix}", "\\begin{pmatrix}", "\\end{pmatrix}", "\\begin{bmatrix}", "\\end{bmatrix}"
    ]

    added_sci = 0
    for sci_tok in math_symbols + physics_chem + latex_commands:
        for pref in ["", " "]:
            tok = f"{pref}{sci_tok}"
            if tok not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE:
                vocab.add_token(tok, tier=TokenTier.WORD if len(tok.split()) <= 1 else TokenTier.PHRASE, frequency=10000, pmi=6.0)
                seen_tokens.add(tok)
                added_sci += 1

    print(f"  • Tier 2/3 MINT-, Physik-, Chemie- & LaTeX-Symbole hinzugefügt: {added_sci:,} Tokens (Stand: {vocab.size:,})")

    # -------------------------------------------------------------------------
    # 7c. Comprehensive Coding, Languages, Systems, Frameworks & DevOps
    # -------------------------------------------------------------------------
    coding_patterns = [
        # Python Magic, Typing & Decorators
        "__init__", "__call__", "__name__", "__main__", "__str__", "__repr__", "__enter__", "__exit__",
        "__iter__", "__next__", "__len__", "__getitem__", "__setitem__", "__delitem__", "__eq__", "__ne__",
        "__lt__", "__gt__", "__le__", "__ge__", "__hash__", "__slots__", "__all__", "__file__", "__doc__", "__annotations__",
        "@staticmethod", "@classmethod", "@property", "@dataclass", "@override", "@lru_cache(", "@abstractmethod",
        "self.", "cls.", "super().__init__(", "isinstance(", "issubclass(", "hasattr(", "getattr(", "setattr(",
        "Optional[", "Union[", "List[", "Dict[", "Tuple[", "Set[", "Callable[", "Any", "TypeVar(", "Generic[",
        "yield from ", "async def ", "raise ValueError(", "raise TypeError(", "raise RuntimeError(", "raise NotImplementedError(", "raise KeyError(",

        # C / C++ & Systems Programming
        "std::vector<", "std::string", "std::unique_ptr<", "std::shared_ptr<", "std::make_unique<", "std::make_shared<",
        "std::move(", "std::forward<", "std::cout << ", "std::endl;", "std::cin >> ", "std::unordered_map<", "std::unordered_set<",
        "std::pair<", "std::tuple<", "std::optional<", "constexpr ", "noexcept ", "nullptr", "const auto& ",
        "size_t", "int32_t", "int64_t", "uint32_t", "uint64_t", "uint8_t", "int8_t", "uint16_t", "int16_t", "ptrdiff_t", "uintptr_t",
        "#include <iostream>", "#include <vector>", "#include <string>", "#include <memory>", "#include <algorithm>", "#include <cmath>", "#include <cstdint>", "#include <thread>", "#include <mutex>",
        "#pragma once", "using namespace std;",

        # Rust Systems Language
        "String::from(", "println!(", "eprintln!(", "format!(", "vec![", "panic!(", "assert_eq!(", "assert_ne!(",
        "Option<", "Result<", "Some(", "None", "unwrap()", "expect(", "pub(crate) ", "pub(super) ",
        "#[derive(Debug, Clone)]", "#[derive(Serialize, Deserialize)]", "#[inline]", "#[tokio::main]",
        "Box<", "Rc<", "Arc<", "RefCell<", "Mutex<", "RwLock<", "AtomicBool", "AtomicUsize", "AtomicU64",
        "&mut self", "&self", "mut ", "impl Trait", "std::sync::Arc", "std::sync::Mutex",

        # JavaScript / TypeScript / React & Web
        "const [", "] = useState(", "useEffect(() => {", "useCallback(", "useMemo(", "useRef(", "useContext(", "useReducer(",
        "export default ", "export const ", "export function ", "export type ", "export interface ", "export class ",
        "Promise<", "Promise.all(", "Promise.resolve(", "Promise.reject(",
        "console.log(", "console.error(", "console.warn(", "document.getElementById(", "document.querySelector(", "addEventListener(",
        "JSON.stringify(", "JSON.parse(", "localStorage.getItem(", "localStorage.setItem(",
        "as const", "as any", "?.", "??", "===", "!==", "=> {",

        # Java & Kotlin Enterprise
        "System.out.println(", "System.err.println(", "throw new IllegalArgumentException(", "throw new NullPointerException(", "throw new IllegalStateException(",
        "@Autowired", "@Entity", "@Table", "@Id", "@GeneratedValue",
        "StringBuilder ", "ArrayList<", "HashMap<", "ConcurrentHashMap<",
        "CompletableFuture.supplyAsync(", "Thread.sleep(", "Executors.newVirtualThreadPerTaskExecutor()",

        # Go Concurrency & Backend
        "func (", ") error {", "if err != nil {", "return nil, err", "go func() {", "sync.Mutex", "sync.RWMutex", "sync.WaitGroup",
        "make(chan ", "make([]", "context.Context", "context.Background()", "fmt.Println(", "fmt.Sprintf(", "json.Unmarshal(", "json.Marshal(",

        # SQL & Database Management
        "INSERT INTO ", "DELETE FROM ", "DROP TABLE IF EXISTS ", "ALTER TABLE ", "PRIMARY KEY (", "FOREIGN KEY (", "CREATE INDEX ",
        "LEFT JOIN ", "INNER JOIN ", "RIGHT JOIN ", "FULL OUTER JOIN ", "CASCADE",

        # AI, Deep Learning & PyTorch
        "import torch\n", "import torch.nn as nn\n", "import torch.nn.functional as F\n",
        "torch.Tensor", "torch.zeros(", "torch.ones(", "torch.cat(", "torch.stack(", "torch.matmul(", "torch.no_grad():",
        "optimizer.step()", "optimizer.zero_grad()", "loss.backward()", "model.eval()", "model.train()",
        "nn.Module", "nn.Linear(", "nn.Embedding(", "nn.LayerNorm(", "nn.Dropout(", "nn.Conv2d(", "nn.MultiheadAttention(",
        "AutoTokenizer.from_pretrained(", "AutoModelForCausalLM.from_pretrained(", "model.generate(",

        # DevOps, Shell & CLI
        "#!/usr/bin/env bash", "set -euo pipefail", "grep -rnI ", "chmod +x ", "chown -R ",
        "docker run -d ", "docker-compose up -d", "docker build -t ", "kubectl get pods", "kubectl apply -f ",
        "git add .", "git commit -m \"", "git checkout -b ", "git push origin ", "git pull --rebase ",
        "npm run build", "npm run dev", "pnpm install", "cargo build --release", "pip install -r requirements.txt", "pytest -v",

        # Formatting & Indentation Lines
        "    return ", "        return ", "            return ",
        "    if ", "        if ", "            if ",
        "    for ", "        for ",
        "    def ", "    class "
    ]

    added_code = 0
    for code_tok in coding_patterns:
        for pref in ["", " "]:
            tok = f"{pref}{code_tok}"
            if tok not in seen_tokens and vocab.size < TARGET_VOCAB_SIZE:
                vocab.add_token(tok, tier=TokenTier.WORD if len(tok.split()) <= 1 else TokenTier.PHRASE, frequency=20000, pmi=7.0)
                seen_tokens.add(tok)
                added_code += 1

    print(f"  • Tier 2/3 Coding-, Framework-, Concurrency- & DevOps-Tokens hinzugefügt: {added_code:,} Tokens (Stand: {vocab.size:,})")

    # -------------------------------------------------------------------------
    # 8. Reserved Standard Protocol Expansion Slots up to exactly 1,048,576
    # -------------------------------------------------------------------------
    res_idx = 0
    while vocab.size < TARGET_VOCAB_SIZE:
        res_tok = f"<reserved_slot_{res_idx:06d}>"
        if res_tok not in seen_tokens:
            vocab.add_token(res_tok, tier=TokenTier.TEMPLATE, frequency=1, pmi=0.0)
            seen_tokens.add(res_tok)
        res_idx += 1

    print(f"  • Tier 3 Reservierte Puffer-Slots allokiert: {res_idx:,} Slots")
    print("=" * 85)
    print(f"✅ EXAKTE 20-BIT VOKABULAR-GRÖSSE ERREICHT: {vocab.size:,} Tokens ({TARGET_VOCAB_SIZE:,})")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # 9. Speichern als JSON & schnelles Binärformat
    # -------------------------------------------------------------------------
    print(f"💾 Speichere kanonisches 20-Bit JSON -> {OUTPUT_JSON}...", flush=True)
    vocab.save_json(OUTPUT_JSON)
    json_bytes = os.path.getsize(OUTPUT_JSON)
    print(f"  • JSON gespeichert: {json_bytes / (1024*1024):.2f} MB")

    print(f"⚡ Speichere ultra-schnelles Binärformat -> {OUTPUT_BIN}...", flush=True)
    bin_hash = save_binary_vocab(vocab, OUTPUT_BIN)
    bin_bytes = os.path.getsize(OUTPUT_BIN)
    print(f"  • Binärdatei gespeichert: {bin_bytes / (1024*1024):.2f} MB")

    # Hash des JSON berechnen
    json_hasher = hashlib.sha256()
    with open(OUTPUT_JSON, "rb") as f:
        while chunk := f.read(65536):
            json_hasher.update(chunk)
    json_hash = json_hasher.hexdigest()

    print("=" * 85)
    print("🔒 KRYPTOGRAFISCHE FINGERPRINTS DES KANONISCHEN 20-BIT VOKABULARS:")
    print(f"  • JSON SHA-256:  {json_hash}")
    print(f"  • Binary SHA-256: {bin_hash}")
    print(f"  • Dauer: {time.time() - start_time:.1f}s")
    print("=" * 85)

    # Metadaten-Hash speichern
    meta_path = "/home/benjamin/Bilder/data/vocab_1m_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": TARGET_VOCAB_SIZE,
            "bit_width": 20,
            "json_sha256": json_hash,
            "binary_sha256": bin_hash,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "languages_covered": "175+ ISO languages (204 NLLB scripts + 61 HermitDave dictionaries)"
        }, f, indent=2)

    return json_hash, bin_hash


if __name__ == "__main__":
    build_1m_20bit_vocabulary()
