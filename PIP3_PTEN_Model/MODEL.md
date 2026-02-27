# PIP3–PTEN モデル（`PIP3_PTEN_Model`）の各項の説明

このフォルダの `nullcline.py`, `dynamics.py`, `dynamics_noise.py` で共通に使っている **空間一様（well-mixed）ODE/SDE** の PIP3–PTEN モデルについて、変数・保存量・各項（生成/分解/結合/解離/正帰還/漏れ/時間スケール/ノイズ）を項別に説明します。

## 目次

- [1. 変数と保存量](#1-変数と保存量)
- [2. 支配方程式（決定論）](#2-支配方程式決定論)
  - [2.1 PIP3（u）の式](#21-pip3uの式)
  - [2.2 PTEN（v）の式](#22-ptenvの式)
- [3. 正帰還 A(u) と、PIP3依存の基礎結合 k_on0(u)](#3-正帰還-au-とpip3依存の基礎結合-k_mathrmon0u)
  - [3.1 PIP3 正帰還 A(u)](#31-pip3-正帰還-au)
  - [3.2 基礎結合の PIP3 依存 k_on0(u)](#32-基礎結合の-pip3-依存-k_mathrmon0u)
- [4. ノイズ付き（確率過程）](#4-ノイズ付き確率過程-dynamics_noisepy)
- [5. ヌルクライン・固定点・興奮性の見方](#5-ヌルクライン固定点興奮性の見方)
- [6. パラメータ対応表](#6-パラメータコードの-par-辞書対応表)
- [7. スクリプトの役割](#7-スクリプトの役割このフォルダ内)
- [8. 参考文献（抜粋）](#8-参考文献抜粋)

---

## 1. 変数と保存量

- **$u$**: 膜上の **PIP3** 濃度（または無次元量）
- **$v$**: **膜結合 PTEN** 量（膜上の PTEN）
- **$p_2$**: 膜上の **PIP2** 量（モデル内では保存量から計算）

このモデルでは膜上の PIP2 と PIP3 の総量を

$$
P_{\mathrm{tot}} = p_2 + u
$$

と置き、常に

$$
p_2 = P_{\mathrm{tot}} - u
$$

で計算します（= **PIP2↔PIP3 変換**として扱う、という仮定）。

また PTEN については、膜結合とそれ以外（細胞質など）の“プール”をまとめて

$$
V_{\mathrm{tot}} = v + (V_{\mathrm{tot}} - v)
$$

とし、膜から外れている量が **$(V_{\mathrm{tot}} - v)$** です。

---

## 2. 支配方程式（決定論）

### 2.1 PIP3（$u$）の式

`nullcline.py` 等で使っている式は次です（コードでは `du_dt` / `f_pip3`）。

$$
\frac{du}{dt}
= \frac{1}{\tau_u}
\left(
v_{\mathrm{PI3K}}\,A(u)\,\frac{p_2}{K_{\mathrm{PI3K}}+p_2}
- v_{\mathrm{PTEN}}\,v\,\frac{u}{K_{\mathrm{PTEN}}+u}
- k_{\mathrm{leak}}\,u
\right)
$$

各項の意味は以下です。

- **(i) PI3K による生成項**  
  - $p_2/(K_{\mathrm{PI3K}}+p_2)$ は **PIP2 を基質**とする飽和型（Michaelis–Menten型）の生成を表します。  
  - $v_{\mathrm{PI3K}}$ は生成の最大速度スケール（強いほど PIP3 が作られやすい）。  
  - $A(u)$ は **PIP3 側の正帰還**で、PIP3 が増えるほど PI3K 反応が実効的に増幅する、という仮定です（次節）。

- **(ii) PTEN による分解項**  
  - $u/(K_{\mathrm{PTEN}}+u)$ は PIP3 を基質とする飽和型の分解（Michaelis–Menten型）です。  
  - 係数として **膜結合 PTEN $v$** が掛かっているので、膜にいる PTEN が多いほど分解が強くなります。  
  - $v_{\mathrm{PTEN}}$ は分解の最大速度スケール。

- **(iii) 漏れ項**  
  - $k_{\mathrm{leak}}u$ は、PTEN 以外の経路（拡散で膜から逃げる、別酵素、分解など）による一次減衰をまとめて表現したものです。

- **$\tau_u$（時間スケール）**  
  - 全反応を $1/\tau_u$ 倍しているので、$\tau_u$ が小さいほど $u$ の変化が速くなります。  
  - `dynamics.py` では stiff になりやすいので `Radau` を使っています。

### 2.2 PTEN（$v$）の式

（コードでは `dv_dt` / `g_pten`）

$$
\frac{dv}{dt}
 =
k_{\mathrm{on}}^{\mathrm{tot}}(u,p_2)\,(V_{\mathrm{tot}}-v)
- k_{\mathrm{off}}\,v
$$

ここで、結合速度は

$$
k_{\mathrm{on}}^{\mathrm{tot}}(u,p_2) = k_{\mathrm{on0}}(u) + k_{\mathrm{on}}\,p_2
$$

としています。

- **(iv) 結合（on）項**  
  - $(V_{\mathrm{tot}}-v)$ は膜にまだ結合していない PTEN の量で、これが大きいほど膜への流入が増えます。  
  - $k_{\mathrm{on}}\,p_2$ は **PIP2 依存**の膜結合（PIP2 が多いほど PTEN が付きやすい）を表します。  
  - $k_{\mathrm{on0}}(u)$ は **PIP2 非依存の基礎結合**で、さらに $u$（PIP3）依存にしてあります（次節）。

- **(v) 解離（off）項**  
  - $k_{\mathrm{off}}v$ は一次の解離（膜から外れる）です。

---

## 3. 正帰還 $A(u)$ と、PIP3依存の基礎結合 $k_{\mathrm{on0}}(u)$

### 3.1 PIP3 正帰還 $A(u)$

$$
A(u)= 1+\alpha \frac{u^n}{K_A^n+u^n}
$$

- $u$ が小さいと $A(u)\approx 1$（増幅なし）
- $u$ が大きくなると $A(u)\to 1+\alpha$（最大で $1+\alpha$ 倍の増幅）
- $n$ が大きいほどスイッチ的（閾値的）になります

この正帰還があることで、パラメータ領域によって **興奮性（excitable）** な挙動（小さな摂動は戻るが、閾値超えで大きな応答をしてから戻る）が出やすくなります。

#### 3.1.1 A(u) 解釈へのつなぎ方（粗視化としての Hill 関数）

1. **多段階修飾・多部位結合・複数の正帰還を含むモジュールは、実効的な Hill 関数で非常によく近似でき、その有効 Hill 係数 $n_{\mathrm{eff}}$ はステップ数や協同性・フィードバックの強さに比例して大きくなり得る。**  
   - Ferrell & Ha, “Ultrasensitivity Part II” は、多部位リン酸化や正帰還が組み合わさると高い Hill 係数を持つスイッチが自然に現れることを総説している。([PMC4435807](https://pmc.ncbi.nlm.nih.gov/articles/PMC4435807/))  
   - Hoops ら, “Hill coefficients, dose–response curves and allosteric mechanisms” は、複雑なアロステリック／多サイト系の入出力がしばしば単一の Hill 式で要約できることを詳しく議論している。([PMC2816740](https://pmc.ncbi.nlm.nih.gov/articles/PMC2816740/))

2. **PIP3 ネットワークでは、実験的に $n_H \sim 3$–$8$ 程度の強いウルトラセンシティビティが観測されているが、支配方程式レベルでは多段階一次反応と複数のフィードバックから構成されている。**  
   - Karunarathne ら, “Optical control demonstrates switch-like PIP3 dynamics underlying the initiation of immune cell migration” では、光刺激に対する PIP3 応答を Hill 式にフィットすると、細胞ごとに $n_H \sim 3$–$8$ の高い有効 Hill 係数が得られる一方、モデルの方程式は質量作用則と複数モジュールから成ることが示されている。([PMC3637758](https://pmc.ncbi.nlm.nih.gov/articles/PMC3637758/))

3. **したがって、本モデルの $A(u)=1+\alpha \frac{u^n}{K_A^n+u^n}$ は、「実際の細胞内で多段階協同性と正帰還が積み重なって実現している急峻なスイッチ挙動」を、PIP3 濃度 $u$ だけの有効 Hill 関数として粗視化したもの、と解釈するのが自然である。**  
   - 上の一般論（多段階系の Hill 近似）と、PIP3 系で観測される高い有効 Hill 係数の両方と整合的。

### 3.2 基礎結合の PIP3 依存 $k_{\mathrm{on0}}(u)$

`history.md` にある通り、このプロジェクトでは v-ヌルクラインの形（傾き）を整えるために、仮説的メカニズムとして

$$
k_{\mathrm{on0}}(u)=k_{\mathrm{on0,base}}
\left(1+\gamma \frac{u}{K_u+u}\right)
$$

を採用しています（コード上のパラメータ名は `Ku_k_on0`）。

- $u$ が増えると $k_{\mathrm{on0}}(u)$ が増える（飽和型）
- $\gamma$ が大きいほど増加の幅が大きい
- $K_u$ は半飽和定数

---

## 4. ノイズ付き（確率過程, `dynamics_noise.py`）

`dynamics_noise.py` では上の決定論的な右辺 $f(u,v), g(u,v)$ に対して、ホワイトノイズ（Wiener過程）を加えた SDE を Euler–Maruyama 法で解いています：

$$
du = f(u,v)\,dt + \sigma_u\,dW_u,\qquad
dv = g(u,v)\,dt + \sigma_v\,dW_v
$$

- $\sigma_u$: PIP3 へのノイズ強度（発火頻度を大きく左右）
- $\sigma_v$: PTEN へのノイズ強度（現在は 0 をよく使う）

実装では数値的な発散や物理的不整合を避けるため、各ステップで
$u\in[0,P_{\mathrm{tot}}]$, $v\in[0,V_{\mathrm{tot}}]$ にクリップしています。

---

## 5. ヌルクライン・固定点・興奮性の見方

- **u-ヌルクライン**: $du/dt=0$ を満たす $(u,v)$ の集合（`compute_nullclines` 内の `v_u`）

$$
v = \frac{v_{\mathrm{PI3K}} A(u)\frac{p_2}{K_{\mathrm{PI3K}}+p_2} - k_{\mathrm{leak}}u}{v_{\mathrm{PTEN}}\frac{u}{K_{\mathrm{PTEN}}+u}}
$$

- **v-ヌルクライン**: $dv/dt=0$ を満たす $(u,v)$ の集合（`v_v`）

$$
v = V_{\mathrm{tot}}\frac{k_{\mathrm{on}}^{\mathrm{tot}}(u,p_2)}{k_{\mathrm{off}}+k_{\mathrm{on}}^{\mathrm{tot}}(u,p_2)}
$$

これらの交点が固定点です。固定点が複数ある（典型的には低 $u$ の安定点＋中間の不安定点など）と、**閾値**を持つ興奮系として振る舞いやすくなります。

---

## 6. パラメータ（コードの `par` 辞書）対応表

`nullcline.py` / `dynamics.py` / `dynamics_noise.py` の `par = dict(...)` にある主要パラメータの意味です。

- **`Ptot`**: $P_{\mathrm{tot}}$（膜上 PIP2+PIP3 総量）
- **`Vtot`**: $V_{\mathrm{tot}}$（PTEN の総量パラメータ）
- **`vPI3K`**: $v_{\mathrm{PI3K}}$（PI3K生成の最大速度係数）
- **`KPI3K`**: $K_{\mathrm{PI3K}}$（PI3K項の飽和定数）
- **`vPTEN`**: $v_{\mathrm{PTEN}}$（PTEN分解の最大速度係数）
- **`KPTEN`**: $K_{\mathrm{PTEN}}$（PTEN項の飽和定数）
- **`k_leak`**: $k_{\mathrm{leak}}$（漏れ）
- **`alpha`**: $\alpha$（正帰還の強さ）
- **`KA`**: $K_A$（正帰還の半飽和）
- **`hill_n`**: $n$（Hill係数）
- **`kon`**: $k_{\mathrm{on}}$（PIP2依存の膜結合係数）
- **`koff`**: $k_{\mathrm{off}}$（膜解離係数）
- **`k_on0_base`**: $k_{\mathrm{on0,base}}$（基礎結合の基準値）
- **`gamma`**: $\gamma$（PIP3依存で基礎結合が増える強さ）
- **`Ku_k_on0`**: $K_u$（基礎結合増加の半飽和）
- **`tau_u`**: $\tau_u$（PIP3の時間スケール）

---

## 7. スクリプトの役割（このフォルダ内）

- **`nullcline.py`**: ヌルクライン・ベクトル場・スライダーでのパラメータ探索
- **`dynamics.py`**: 決定論 ODE の時間発展（閾値以下/以上の比較）
- **`dynamics_noise.py`**: ノイズ駆動 SDE（自発発火の再現）

---

## 8. 参考文献（抜粋）

- Ferrell & Ha, “Ultrasensitivity Part II”. `https://pmc.ncbi.nlm.nih.gov/articles/PMC4435807/`
- Hoops ら, “Hill coefficients, dose–response curves and allosteric mechanisms”. `https://pmc.ncbi.nlm.nih.gov/articles/PMC2816740/`
- Karunarathne ら, “Optical control demonstrates switch-like PIP3 dynamics underlying the initiation of immune cell migration”. `https://pmc.ncbi.nlm.nih.gov/articles/PMC3637758/`

