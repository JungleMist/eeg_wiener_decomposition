# ECMAD 相位门控 Wiener 分解

ECMAD 在预先定义的 EEG 通道组内，用同组的其他通道来预测目标通道的信号。相位门控版（`phasegated`）先算出 Wiener 预测系数，再逐个判断参考通道在各个频点是否和目标通道足够接近同相。通过门控的预测项相加得到 **coherent（可预测成分）**，从原信号减去它得到 **specific（残差成分）**。

$$
x_{\mathrm{raw}}(t)=x_{\mathrm{coherent}}(t)+x_{\mathrm{specific}}(t)
$$

## 1. 相位阈值怎样工作

设目标通道为 $y$，同组某个参考通道为 $r$，两者在频点 $f$ 的互功率谱密度记作 $S_{yr}(f)$，简称互谱。互谱是一个复数，它的幅角 $\arg S_{yr}(f)$ 表示两通道在该频点估计出的相对相位。两路波形在该频率下的波峰大致对齐时，相位差接近 $0$。一路的波峰对上另一路的波谷时，相位差的绝对值接近 $\pi$。

$$
\begin{aligned}
d_r(f) &= \left|\arg S_{yr}(f)\right|, \qquad d_r(f)\in[0,\pi],\\
g_r(f) &= \begin{cases}
1, & d_r(f)\le\theta,\\
0, & d_r(f)>\theta,
\end{cases}
\qquad
\widetilde h_r(f)=g_r(f)h_r(f).
\end{aligned}
$$

$\theta$ 是允许的最大相位差，单位为弧度，配置项为 `wiener.phase_gate_threshold_rad`。相位角落在闭区间 $[-\theta,+\theta]$ 内时，原复数 Wiener 系数 $h_r(f)$ 直接通过。超出区间时，该参考通道在这个估计频点的系数置零。取绝对值使正、负相位差遵循同一规则。

| 阈值 | 含义 |
|---|---|
| $0$ | 只让计算得到的相位差恰好为零的频点通过 |
| $0.20$ rad（约 $11.46^\circ$） | 接受 $[-0.20,+0.20]$ rad，包括两个端点 |
| $\pi$（$180^\circ$） | 所有相位通过，包括反相。其余设置相同时，分解结果与普通复数 Wiener（`frequency`）一致 |

例如，在 25 Hz 处，目标通道 T3 的两个参考通道 T5 和 O1 的互谱相位分别为 $-0.10$ 和 $+0.60$ rad。设 $\theta=0.20$ rad，则 T5 的预测项通过，O1 的预测项置零。在其他频点，两条参考通道各自重新判断。仓库的 `configs/default.yaml` 将阈值设为 $0.050$ rad（约 $2.86^\circ$），这里的 $0.20$ rad 用于演示。 [相位门控实现](../eeg_bg/decomposition/phase_gate.py) · [默认配置](../configs/default.yaml)

## 2. Wiener 系数怎样求出

记 $X_i(f)$ 为通道 $i$ 在频点 $f$ 的傅里叶系数。ECMAD 用 Welch 方法估计互谱矩阵 $S(f)$，即把信号分段，对各段的频域乘积取平均并按谱密度归一化。按 SciPy 的约定，乘积取 $X_i(f)^*X_j(f)$，其中 $*$ 表示复共轭。矩阵对角线 $S_{ii}$ 是各通道的功率谱密度，非对角线 $S_{ij}$ 是通道之间的互谱。 [SciPy 互谱定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html) · [ECMAD 互谱估计](../eeg_bg/decomposition/wiener.py)

用 $R$ 表示所有参考通道，$S_{RR}(f)$ 是参考通道之间的互谱矩阵，$S_{Ry}(f)$ 是参考通道到目标通道的互谱列向量。每个频点解一次线性方程，得到一组复数系数：

$$
\bigl(S_{RR}(f)+\lambda(f)I\bigr)\mathbf h(f)=S_{Ry}(f).
$$

$I$ 是单位矩阵。$\lambda(f)$ 是按参考通道平均功率谱密度缩放的小正数，用来稳定求解，尤其是参考通道彼此高度相关时。只有一个参考通道时，公式就是 $h_r(f)=S_{ry}(f)/(S_{rr}(f)+\lambda(f))$。多个参考通道一起参与求解，得到的系数随后逐个乘以相位门控 $g_r(f)$。

## 3. 怎样得到两路波形

将各参考通道的频域信号乘上门控后的系数，再相加。逆傅里叶变换后得到时域的可预测成分，剩下的就是目标通道的残差。

$$
\begin{aligned}
\widehat Y_{\mathrm{coherent}}(f)
  &= \sum_{r\in R}\widetilde h_r(f)X_r(f),\\
x_{\mathrm{coherent}}(t)
  &= \mathcal F^{-1}\!\left[\widehat Y_{\mathrm{coherent}}\right](t),\\
x_{\mathrm{specific}}(t)
  &= x_{\mathrm{raw}}(t)-x_{\mathrm{coherent}}(t).
\end{aligned}
$$

公式描述单个通道组中的一个目标通道。代码在 Welch 估计网格上执行相位门控。如果重建使用更密的 FFT 网格，则先插值门控后的系数。完整流程还受功率门限、相干性门限、保护频带和重叠通道组融合影响。 [相位门控模式](../eeg_bg/decomposition/wiener_phasegated.py) · [滤波重建与融合](../eeg_bg/decomposition/wiener.py)

相位门控依赖一个工作假设，即部分共享伪迹会在组内表现为近同相。近同相的真实脑活动也可能进入 `coherent`，而相位差较大的伪迹可能留在 `specific`。阈值变宽会让更多参考通道与频点组合通过。实际减去多少信号还取决于各预测项的幅度，以及它们相加时是增强还是抵消。
