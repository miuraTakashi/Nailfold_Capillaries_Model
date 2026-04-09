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
  - $k_{\mathrm{off}}(u)\,v$ は一次の解離で、$k_{\mathrm{off}}(u)$ を $u$ の関数にできる（オプション）。  
  - Matsuoka らは PIP3 が高いほど PTEN の膜解離が速くなることを示しており、  
    $k_{\mathrm{off}}(u) = k_{\mathrm{off}}\,\bigl(1 + \delta_{\mathrm{off}}\, u/(K_{u,\mathrm{off}}+u)\bigr)$ とするとその傾向を反映できる（`delta_off=0` のときは従来どおり定数 $k_{\mathrm{off}}$）。

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

### 3.2 基礎結合の PIP3 依存 $k_{\mathrm{on0}}(u)$（Matsuoka 型相互抑制に合わせた改訂）

Matsuoka ら ([Nat. Commun. 9, 4481 (2018)](https://www.nature.com/articles/s41467-018-06856-0)) は、**PIP3 が PTEN の膜局在を負に制御する**（PIP3 が高いと PTEN の膜結合が抑えられる）ことを示しています。  
この「PIP3–PTEN の相互抑制（mutual inhibition）」を粗視化するため、本モデルでは

$$
k_{\mathrm{on0}}(u)
=
\frac{k_{\mathrm{on0,base}}}{1+\gamma \frac{u}{K_u+u}}
$$

という **単調減少型** の基礎結合速度を採用しました（コード上のパラメータ名は `k_on0_base`, `gamma`, `Ku_k_on0`）。

- $u$ が小さいとき $k_{\mathrm{on0}}(u)\approx k_{\mathrm{on0,base}}$（PIP3 による抑制は弱い）
- $u$ が大きくなると $k_{\mathrm{on0}}(u)\to k_{\mathrm{on0,base}}/(1+\gamma)$ に飽和し、**PIP3 が高い領域ほど PTEN の PIP2 非依存基礎結合が弱くなる**  
- $\gamma$ が大きいほど、PIP3 による結合抑制の強さが増す
- $K_u$ は「どの程度の $u$ で抑制が効き始めるか」を決める半飽和定数

PIP3 による抑制の分子実体（どの脂質・結合部位が失活するか）は明示的にはモデル化しておらず、
Matsuoka らが示した

- PIP3 が PTEN の「安定な膜結合状態」の割合を減らす  
- 高 PIP3 では PTEN の膜解離が速くなる（実効的に膜局在が抑えられる）

という現象を、**「PIP3 が高いほど、PIP2 非依存の基礎 on-rate が低下する」** という 1 次元の粗視化で表現しています。

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

- **v-ヌルクライン**: $dv/dt=0$ を満たす $(u,v)$ の集合（`v_v`）。$k_{\mathrm{off}}$ を $u$ 依存にした場合は $k_{\mathrm{off}}(u)$。

$$
v = V_{\mathrm{tot}}\frac{k_{\mathrm{on}}^{\mathrm{tot}}(u,p_2)}{k_{\mathrm{off}}(u)+k_{\mathrm{on}}^{\mathrm{tot}}(u,p_2)}
$$

これらの交点が固定点です。固定点が複数ある（典型的には低 $u$ の安定点＋中間の不安定点など）と、**閾値**を持つ興奮系として振る舞いやすくなります。

#### 5.1 興奮性について（Matsuoka 型改訂後）

Matsuoka 型に改訂した現状の方程式系（$k_{\mathrm{on0}}(u)$ が $u$ で単調減少、およびオプションで $k_{\mathrm{off}}(u)$ が $u$ で増加）では、**v-ヌルクラインが $u$ に対して単調減少**するため、u-ヌルクラインとの交点は **1 つ** になりがちです。固定点が 1 つだけのときは、その点が安定ノードとなり、閾値や「小さな摂動は戻り・閾値超えで大きな応答」という**興奮性（excitability）**は現れません。

興奮系にするには、**固定点を 2 つ以上**（例：低 $u$ の安定点＝静止状態と、中間 $u$ のサドル＝閾値）にする必要があります。そのためには、v-ヌルクラインが $u$ のある区間で**正の傾き**を持ち、u-ヌルクラインと複数回交わる形が望ましいです。しかし、

- $k_{\mathrm{on}}^{\mathrm{tot}}(u,p_2)$ は $u$ の増加で減少（PIP2 減少と $k_{\mathrm{on0}}(u)$ の減少）
- $k_{\mathrm{off}}(u)$ を Matsuoka に合わせて $u$ で増加させると、$v = V_{\mathrm{tot}} k_{\mathrm{on}}^{\mathrm{tot}}/(k_{\mathrm{off}}+k_{\mathrm{on}}^{\mathrm{tot}})$ の分母がより増え、v-ヌルクラインはやはり $u$ で単調減少

となるため、**「PIP3 が PTEN を抑制する」という符号だけを忠実に反映すると、この 2 変数系だけでは興奮性は出にくい**です。

取りうる対応は次のとおりです。

1. **パラメータ探索**  
   `nullcline.py` のスライダーで `gamma`, `alpha`, `KA`, `hill_n`, `delta_off`, `Ku_k_off` などを変え、u-ヌルクラインの形と v-ヌルクラインの位置の組み合わせで、複数交点が出る領域がないか探す。

2. **解離の PIP3 依存 $k_{\mathrm{off}}(u)$ の追加**  
   コード上では $k_{\mathrm{off}}(u) = k_{\mathrm{off}}\,\bigl(1 + \delta_{\mathrm{off}}\, u/(K_{u,\mathrm{off}}+u)\bigr)$ を入れてあり（`delta_off`, `Ku_k_off`）、Matsuoka の「PIP3 が解離を促進する」を反映できます。ただしこのだけでは v-ヌルクラインは依然単調減少になることが多く、固定点が 1 つのままになる場合があります。

3. **興奮性を目的とした追加項（現象論）**  
   興奮性を確保するために、v-ヌルクラインに「中程度の $u$ でわずかに上にふくらむ」ような項を現象論的に追加する方法があります。その場合は、MODEL 上で「Matsuoka の定性的な符号とは別に、閾値構造を得るための補正」と明記する必要があります。

まとめると、**現行の Matsuoka 型（$k_{\mathrm{on0}}(u)$ 減少＋任意で $k_{\mathrm{off}}(u)$ 増加）のままだと、この方程式系は興奮系にならない**ことが多く、興奮性を得るには上記のいずれか（パラメータ探索・追加の u 依存・現象論的補正）を検討する必要があります。

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
- **`gamma`**: $\gamma$（PIP3 による基礎結合“抑制”の強さ）
- **`Ku_k_on0`**: $K_u$（基礎結合抑制の半飽和）
- **`delta_off`**: $\delta_{\mathrm{off}}$（PIP3 による解離促進の強さ。0 で $k_{\mathrm{off}}$ は定数）
- **`Ku_k_off`**: $K_{u,\mathrm{off}}$（解離促進の半飽和）
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

---

## 9. `u` 消去モデル（`w,v` の2変数系）の方程式

対象ファイル:

- `RASA1_PTEN_with_eliminated_PIP3.py`
- `RASA1_PTEN_with_eliminated_PIP3_swapped_taus.py`

両者とも、まず高速変数 $u$（PIP3）を準定常近似で消去します。

$$
u = \hat u(v,w), \qquad \text{where } \frac{du}{dt}(u,v,w)=0
$$

`u_hat(v,w)` は次の**陰的方程式**の根として定義されます（コードの `u_hat` 関数）:

$$
0=
\frac{1}{\tau_u}
\left[
v_{\mathrm{PI3K}}\,A(u)\,\frac{P_{\mathrm{tot}}-u}{K_{\mathrm{PI3K}}+(P_{\mathrm{tot}}-u)}
\cdot \frac{1}{1+\beta_w w}
- v_{\mathrm{PTEN}}\,v\,\frac{u}{K_{\mathrm{PTEN}}+u}
- k_{\mathrm{leak}}u
\right]
$$

$$
A(u)=1+\alpha\frac{u^n}{K_A^n+u^n},\qquad
u\in[0,P_{\mathrm{tot}}]
$$

したがって `u_hat(v,w)` は一般に閉形式ではなく、各 $(v,w)$ ごとに数値的に求めます。
実装では区間走査 + 2分法で候補根を求め、$\partial(du/dt)/\partial u<0$ の安定枝を優先しています。

このとき、状態変数を $(w,v)$ とした 2 変数系は次です。

$$
\frac{dw}{dt}
=
\frac{w_{\infty}(\hat u(v,w)) - w}{\tau_w}
$$

$$
\frac{dv}{dt}
=
\frac{1}{\tau_v}
\left[
\left(k_{\mathrm{on0}}(\hat u(v,w)) + k_{\mathrm{on}}(P_{\mathrm{tot}}-\hat u(v,w))\right)(V_{\mathrm{tot}}-v)
- k_{\mathrm{off}}(\hat u(v,w))\,v
\right]
$$

ここで

$$
w_{\infty}(u)=w_{\max}\frac{u^m}{K_w^m+u^m},
\qquad
k_{\mathrm{on0}}(u)=\frac{k_{\mathrm{on0,base}}}{1+\gamma\,u/(K_{u,\mathrm{on0}}+u)}
$$

で、$k_{\mathrm{off}}(u)$ は

- 定数版: $k_{\mathrm{off}}(u)=k_{\mathrm{off}}$
- 拡張版: $k_{\mathrm{off}}(u)=k_{\mathrm{off}}\left(1+\delta_{\mathrm{off}}\,u/(K_{u,\mathrm{off}}+u)\right)$

を使います（コードの `delta_off` で切替）。

### 9.1 通常版（`RASA1_PTEN_with_eliminated_PIP3.py`）

現在のデフォルト値:

- $\tau_w = 0.3$
- $\tau_v = 6.0$

すなわち、`w`（intermediate）より `v`（slow）を遅く設定しています。

### 9.2 `tau` 交換版（`RASA1_PTEN_with_eliminated_PIP3_swapped_taus.py`）

`swapped_taus` 版では時間スケールのみ入れ替えます。

$$
\tau_w \leftrightarrow \tau_v
$$

（他の方程式・パラメータは同一）

---

## 10. パラメータ表（`u` 消去モデル）

基準: `RASA1_PTEN_with_eliminated_PIP3.py` の `par`。  
`RASA1_PTEN_with_eliminated_PIP3_swapped_taus.py` では **`tau_w` と `tau_v` のみ交換**します。

| コード名 | 記号 | 生物学的意味（モデル上の解釈） | 値（基準） |
|---|---|---|---:|
| `Ptot` | \(P_{\mathrm{tot}}\) | 膜上 PIP2+PIP3 の総量（保存量） | 1.000 |
| `vPI3K` | \(v_{\mathrm{PI3K}}\) | PI3K 側の PIP3 生成最大速度係数 | 1.274 |
| `KPI3K` | \(K_{\mathrm{PI3K}}\) | PI3K 生成項の飽和定数 | 0.010 |
| `vPTEN` | \(v_{\mathrm{PTEN}}\) | PTEN 側の PIP3 分解最大速度係数 | 6.810 |
| `KPTEN` | \(K_{\mathrm{PTEN}}\) | PTEN 分解項の飽和定数 | 0.100 |
| `k_leak` | \(k_{\mathrm{leak}}\) | PIP3 の漏れ/背景減衰 | 5.667 |
| `alpha` | \(\alpha\) | PIP3 正帰還の強さ | 5.000 |
| `KA` | \(K_A\) | 正帰還 Hill 関数の半飽和定数 | 0.300 |
| `hill_n` | \(n\) | 正帰還 Hill 係数（非線形性） | 4.021 |
| `kon` | \(k_{\mathrm{on}}\) | PIP2 依存 PTEN 膜結合係数 | 2.000 |
| `koff` | \(k_{\mathrm{off}}\) | PTEN 膜解離係数（基準） | 18.000 |
| `k_on0_base` | \(k_{\mathrm{on0,base}}\) | PIP2 非依存 PTEN 基礎結合の基準値 | 7.143 |
| `Vtot` | \(V_{\mathrm{tot}}\) | PTEN 有効総量（膜結合+可用プール） | 0.984 |
| `gamma` | \(\gamma\) | PIP3 による基礎結合抑制強度 | 13.571 |
| `Ku_k_on0` | \(K_{u,\mathrm{on0}}\) | 基礎結合抑制の半飽和定数 | 0.347 |
| `delta_off` | \(\delta_{\mathrm{off}}\) | PIP3 依存解離促進の強さ（0で無効） | 0.0 |
| `Ku_k_off` | \(K_{u,\mathrm{off}}\) | 解離促進の半飽和定数 | 0.4 |
| `tau_u` | \(\tau_u\) | 高速変数 \(u\) の時定数（準定常化の基準） | 0.010 |
| `beta_w` | \(\beta_w\) | \(w\) による PI3K 生成抑制の強さ | 3.0 |
| `tau_w` | \(\tau_w\) | \(w\)（RASA1-like）の時定数（intermediate） | 0.3 |
| `w_max` | \(w_{\max}\) | \(w_\infty(u)\) の最大値 | 1.0 |
| `Kw` | \(K_w\) | \(w_\infty(u)\) の半飽和定数 | 0.25 |
| `hill_m` | \(m\) | \(w_\infty(u)\) の Hill 係数 | 4.0 |
| `tau_v` | \(\tau_v\) | \(v\)（PTEN）の時定数（slow） | 6.0 |

補足（`swapped_taus` 版）:

- `tau_w = 6.0`
- `tau_v = 0.3`

---

## 11. 薬理で制御しやすいパラメータと試薬の対応

> 注意: 下表は「モデル上の実効パラメータ」と「実験操作」の対応づけです。  
> 実際の効果量・選択性・作用方向は細胞種・濃度・処理時間で変わるため、必ず系ごとに検証してください。

| モデル上の主パラメータ | 想定される操作方向 | 代表的試薬（例） | モデル解釈（どこに効かせるか） |
|---|---|---|---|
| `vPI3K` | ↓ | `LY294002`, `Wortmannin` | PI3K 活性低下として PIP3 生成項を抑制 |
| `vPI3K`（class I寄り） | ↓ | `PIK-90`, `GDC-0941` | class I PI3K 寄りの抑制として扱う |
| `vPTEN` | ↓ | `bpV(HOpic)`, `bpV(phen)` | PTEN 実効活性低下（PIP3 分解能低下） |
| `vPTEN`（酸化失活） | ↓ | `H2O2` | 酸化に伴う PTEN 実効活性低下として近似 |
| `k_leak`（PIP3背景分解の実効） | ↓ | `3AC`（SHIP1）, `AS1949490`（SHIP2） | PTEN 以外の PIP3 分解経路の実効低下 |
| `koff`, `delta_off` | ↑/↓ | （PTEN 膜局在操作; 直接特異薬は限定的） | PTEN 膜解離の実効変化として反映 |
| `k_on0_base`, `gamma` | ↑/↓ | `LY294002`, `Wortmannin`（間接） | PIP3 低下を介した PTEN 膜再結合の実効変化 |
| `beta_w`, `tau_w`, `Kw`（RASA1様負帰還） | ↑/↓ | `Tipifarnib`, `U0126`, `PD0325901`（間接） | Ras/RasGAP 軸の負帰還強度・時定数の実効変化 |

実務上は、まず以下が扱いやすいです。

- PI3K 軸: `LY294002` / `Wortmannin` → 主に `vPI3K`（+間接に `gamma` 側）
- PTEN 軸: `bpV(HOpic)` / `bpV(phen)` → 主に `vPTEN`
- 負帰還軸: Ras/ERK 側阻害剤で `beta_w`, `tau_w` の実効を探索

