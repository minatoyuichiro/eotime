# eotime — Exchange-Only 量子ビットの「時間精度」はアルゴリズムの答えにどう効くか

## 題
*Pulse-timing precision in exchange-only spin qubits: how it propagates to VQE, QAOA and
fault-tolerance primitives*（仮）

## 問い
Exchange-Only（EO）量子ビットは 3 スピンに 1 論理量子ビットを符号化し、**全てのゲートが
交換相互作用パルスの時間列**になる（1 量子ビット回転 = 1〜4 パルス、CNOT = 20 パルス + 補正 8、
論理 SWAP = 9 パルス）。パルスの長さ θ = J·t/ħ の誤差は論理ゲート誤差と**符号空間からの漏れ
（S=3/2 への leakage）**になる。これが代表的アルゴリズムの答えにどう効くかを数値で測る。

## 比べる相手（誤差の種類）
| 記号 | 何の誤差か | 入れ方（t は SWAP 時間 π/J を 1 とした長さ） |
|---|---|---|
| ε | 系統・乗法（J や時計の較正ずれ） | t → t(1+ε)、全パルス共通 |
| δ | 系統・加法（立ち上がり／立ち下がりのオフセット） | t → t+δ、全パルス共通 |
| σ_mult | 揺らぎ・乗法（パルスごと独立） | t → t(1+N(0,σ)) |
| σ_add | 揺らぎ・加法（パルスごと独立） | t → t+N(0,σ) |

## 測り方（1 つは何の 1 つか）
- **ゲート単位**: 論理部分空間への射影ブロック A から、不忠実度 1−|tr(U†A)|²/d² と
  漏れ 1−tr(A†A)/d。揺らぎは 100 回の独立な引き直しの平均。
- **VQE（H2, STO-3G, R=0.735 Å, parity 写像 2 量子ビット）**: 厳密基底 −1.85727503 Ha に対する
  ΔE。期待値は物理的に測る（基底変換もパルス、読み出しはペアの一重項／三重項判定。
  漏れた成分は三重項＝|1⟩ と読まれる）。基準は化学精度 1.6 mHa。θ を固定した場合と、
  誤差下で θ を再最適化した場合の両方。
- **QAOA MaxCut（C4・P6・prism6, p=1,2）**: 近似比 ⟨C⟩/C_max と最適カット確率。
  論理量子ビットは一列に並び、非隣接の辺は論理 SWAP でルーティング（SWAP も誤差を受ける）。
- **FTQC 寄り**: Grover 3 量子ビット（成功確率、理想 0.9453）、QPE 3+1 量子ビット
  （正解確率、φ=3/8 は理想 1.0、φ=0.3 は理想 0.578）、反復符号 d=3 の 1 ラウンド
  （訂正後の論理誤り率と漏れ）。漏れた成分は失敗に数える。
- 測定ショット雑音は入れていない（期待値は厳密）。

## 符号化の規約（v2, 2026-10-12）
|0_L> = |S>_12 ⊗ |↑>_3（前の 2 スピンが一重項）。v1（|↑>_1 ⊗ |S>_23）からの変更と、数値が変わらない理由・検証は `notes/V2_CONVENTION.md`。

## 置き場
- `code/eo_sim.py`   … 3n スピン・Sz 固定セクターの状態ベクトル、パルス、符号化、Zeuch–Bonesteel の
  2 量子ビット列、1 量子ビット合成（4 パルス、総時間最小の解）
- `code/circuit.py`  … 論理回路 → パルス（隣接 1 量子ビットゲートの併合、ルーティング、CCZ、CP）
- `code/gate_errors.py` / `vqe_h2.py` / `qaoa_maxcut.py` / `ftqc.py` … 実験
- `results/*.json` … 数値（`measured_at` に測った時刻）、`results/*.log` … 走らせたときの画面

## 参照
- Fong & Wandzura, QIC 11, 1003 (2011) … 18/22 パルスの CNOT（遺伝的アルゴリズムで発見）
- Zeuch & Bonesteel, PRA 93, 010303(R) (2016) … 上の解析的導出。本コードはこの構成
  （V = U(1/2)U(3/2) を芯にした 20 パルス）をそのまま実装し、数値で漏れ 0・CNOT 忠実度 1 を確認
- 近隣: `~/papers/_sources/eo`（EO ノート、読むだけ）・`~/papers/eoqec`・`~/papers/teex-mnist`

## 変種の結果ディレクトリ
- `results_cz/`, `results_lccz/` … Chadwick の CZ / 漏れ抑制 CZ を公開どおりの向きで使った現行版（論文の表 VIII）。
- `results_cz_v1/`, `results_lccz_v1/` … v1 の符号化で 6 ドット反転＋基底回転 L を使った旧版。論文では使っていない（記録用）。

## arXiv 投稿用パッケージ
- `paper/eotime_arxiv_source.tar.gz` は `python3 code/make_arxiv.py` で作る（`paper/arxiv/` に表を埋め込んだ `main.tex` と `main.bbl`・`refs.bib`・`figs/` を集めて固める）。arXiv の pdflatex は REVTeX の表の中で `\@@input` を使えないため、表は埋め込む。
