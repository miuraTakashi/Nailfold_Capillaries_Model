# coupled_vessel_pip3.py の支配方程式（現行実装）

このファイルは `coupled_vessel_pip3.py` の**現行実装**（PIP3–PTEN で駆動、閾値型 $\eta$、外向き符号付き）に対応する支配方程式をまとめたものです。

## 変数と添字

- 界面（血管壁）: $i\in\{1,2,3,4\}$
- 空間: $x\in[0,L]$（数値計算では格子点 $x_j=j\,\Delta x$）
- 血管壁位置: $h_i(x,t)$
- PIP3 / PTEN: $u_i(x,t),\,v_i(x,t)$
- 静止状態（コードでは数値的に算出）: $(u_*,v_*)$

外向きに押すための符号は
$$
(s_1,s_2,s_3,s_4)=(+1,-1,+1,-1)
$$
（$\eta_2,\eta_4$ を負向きに入れる、に対応）とする。

---

## PIP3–PTEN（各格子点・各界面で独立な確率過程）

各 $(i,x)$ で独立に、次の確率微分方程式（Euler–Maruyamaで離散化）を解く:

$$
\mathrm{d}u_i(x,t)= f\!\big(u_i(x,t),v_i(x,t)\big)\,\mathrm{d}t
 + \sigma_u\,\mathrm{d}W^{(u)}_{i,x}(t)
$$
$$
\mathrm{d}v_i(x,t)= g\!\big(u_i(x,t),v_i(x,t)\big)\,\mathrm{d}t
 + \sigma_v\,\mathrm{d}W^{(v)}_{i,x}(t)
$$

ここで $W^{(u)}_{i,x},W^{(v)}_{i,x}$ は**$(i,x)$ ごとに独立**な標準ブラウン運動。

反応項はコード中の関数に対応し、概略は

$$
f(u,v)=\frac{1}{\tau_u}\left[
v_{\mathrm{PI3K}}\,A(u)\,\frac{(P_{\mathrm{tot}}-u)}{K_{\mathrm{PI3K}}+(P_{\mathrm{tot}}-u)}
-v_{\mathrm{PTEN}}\,v\,\frac{u}{K_{\mathrm{PTEN}}+u}
-k_{\mathrm{leak}}\,u
\right]
$$

$$
g(u,v)=k_{\mathrm{on}}(u)\,(V_{\mathrm{tot}}-v)-k_{\mathrm{off}}\,v
$$

ただし

$$
A(u)=1+\alpha\,\frac{u^n}{K_A^n+u^n},
\quad
k_{\mathrm{on}}(u)=k_{\mathrm{on0}}(u)+k_{\mathrm{on}}^{(2)}(P_{\mathrm{tot}}-u)
$$
$$
k_{\mathrm{on0}}(u)=k_{\mathrm{on0,base}}\left(1+\gamma\,\frac{u}{K_u+u}\right)
$$

（記号は `pip3_par` の各パラメータに対応）。

---

## 血管壁（4界面）の支配方程式

各界面は拡散（表面張力に相当）とバネ力、そして PIP3 由来の駆動で動く:

$$
\partial_t h_i(x,t)
=d_h\,\partial_x^2 h_i(x,t)+F^{\mathrm{spring}}_i(h_1,h_2,h_3,h_4)
\;+\;\sigma_c\,s_i\,\eta_i(x,t).
$$

### 閾値型の駆動（ノイズ相当項）

現行実装では「閾値を超えた分だけ押す」形で

$$
\eta_i(x,t)=\max\!\big(u_i(x,t)-u_{\mathrm{th}},\,0\big)
$$

を用いる（コードでは `u_threshold`。現在の既定値は `0.4`）。

### バネ力（コードの各式に一致）

以下では $H=\frac{w_2}{2}+2r+w_1$。

**界面 1**:
$$
F^{\mathrm{spring}}_1=
-\frac{k_s}{w_1}\left(h_1-(H-w_1)\right)
\;+\;\frac{k_e}{2r}\left(h_2-h_1+2r\right)
$$

**界面 2**:
$$
F^{\mathrm{spring}}_2=
-\frac{k_e}{2r}\left(h_2-h_1+2r\right)
\;+\;\frac{k_s}{w_2}\left(h_3-h_2+w_2\right)
$$

**界面 3**:
$$
F^{\mathrm{spring}}_3=
-\frac{k_s}{w_2}\left(h_3-h_2+w_2\right)
\;+\;\frac{k_e}{2r}\left(h_4-h_3+2r\right)
$$

**界面 4**:
$$
F^{\mathrm{spring}}_4=
-\frac{k_e}{2r}\left(h_4-h_3+2r\right)
\;-\;\frac{k_s}{w_1}\left(h_4+H-w_1\right)
$$

---

## 数値計算上の境界条件・拘束（実装の注意）

- 空間ラプラシアンは `np.roll` による周期差分で計算しているが、毎ステップ
  $$
  h_i(0,t)=h_i(L,t)=h_i^{\mathrm{eq}}
  $$
  として両端を平衡値に固定（コードの `h_eq`）。
- クリップ拘束:
  $$
  h_1(x,t)\le H,\quad h_4(x,t)\ge -H.
  $$

