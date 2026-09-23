# Signal Processing and Tracking Theory

Echorin is an educational radar and sonar simulator. Its algorithms are real,
but its point-target propagation and ideal angular channels are deliberately
simpler than high-fidelity electromagnetic or acoustic models.

## Geometry and two-way propagation

For sensor position $(x_s,y_s)$ and target position $(x_t,y_t)$, Echorin
computes

$$
R=\sqrt{(x_t-x_s)^2+(y_t-y_s)^2},\qquad
\theta=\mathrm{atan2}(y_t-y_s,x_t-x_s).
$$

A monostatic pulse travels to the target and back, so its two-way delay is
$\tau=2R/c$. Radar uses $c=299,792,458$ m/s. The default water sonar
uses $c=1500$ m/s. The sampled echo begins at
$n_d=\mathrm{round}(\tau f_s)$, and range bin $n$ maps back to
$R_n=cn/(2f_s)$.

The current amplitude law is explicitly simplified:

$$
A_r=\rho\left(\frac{R_0}{\max(R,R_0)}\right)^p,
$$

where $\rho$ is reflectivity, $R_0$ is a reference range, and $p=2$ by
default. Additive white Gaussian noise uses either a specified standard
deviation or one derived from requested SNR.

## Waveforms and matched filtering

The rectangular pulse is $s(t)=A\cos(2\pi f_ct)$. The LFM waveform is

$$
s(t)=A\cos\left(2\pi(f_0t+\tfrac12kt^2)\right),\qquad k=B/T_p.
$$

The matched filter correlates received samples with the conjugate time-reversed
transmit waveform. Echorin retains nonnegative lags and aligns output index zero
with zero delay. Pulse compression therefore produces a range peak without
looking at target truth.

## CA-CFAR detection

Cell-Averaging Constant False Alarm Rate (CA-CFAR) estimates noise power from
training cells on both sides of a cell under test (CUT). Guard cells around the
CUT prevent target energy from contaminating that estimate. With $N$ total
training cells and requested false-alarm probability $P_{FA}$, square-law
CA-CFAR uses

$$
\alpha=N(P_{FA}^{-1/N}-1),\qquad T=\alpha\bar P_{noise}.
$$

Echorin compares profile power with $T$, retains local maxima, and exposes
$\sqrt T$ for plotting against matched-filter magnitude.

## Doppler processing

A coherent pulse interval samples target phase at the PRF. Echorin applies an
optional Hann window and FFT along slow time for every range bin. For wavelength
$\lambda=c/f_c$, monostatic Doppler and radial velocity obey

$$
f_d=\frac{2v_r}{\lambda},\qquad v_r=\frac{f_d\lambda}{2}.
$$

The velocity resolution is $\Delta v=(\mathrm{PRF}/N_p)\lambda/2$, and the
unambiguous interval follows $|f_d|<\mathrm{PRF}/2$. Echorin rejects simulated
targets outside that interval instead of silently aliasing them.

### Range-Doppler image coordinates

The same coherent FFT produces a complex spectrum $S[k,n]$ for every velocity
bin $k$ and matched-filter range bin $n$. Echorin stores the complete product in
`[velocity, range]` order. The horizontal cell centers are
$R_n=cn/(2f_s)$ metres; the vertical centers are
$v_k=f_k\lambda/2$ metres per second, where $f_k$ is the shifted slow-time FFT
frequency. Positive velocity means increasing sensor-target range.

The heatmap offers linear magnitude $|S[k,n]|$ or unit-reference amplitude dB:

$$
L_{\mathrm{dB}}[k,n]=\max\left(-100,20\log_{10}|S[k,n]|\right).
$$

Zero and non-finite display values are assigned the $-100$ dB floor. This
display transform does not alter the FFT, detector, or estimated velocity. Red
markers show detections at their measured range and radial velocity; selecting
a cell reports its physical coordinates, magnitude, dB level, and bin indices.

## Kalman tracking and association

Tracks use state $\mathbf{x}=[x,y,v_x,v_y]^T$ and transition

$$
F = 
\begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

The process covariance is the discretized white-acceleration model. Position
updates use a linear measurement matrix and a Joseph-form covariance correction
for numerical stability. Greedy nearest-neighbor association sorts all gated
Mahalanobis distances and enforces one detection and one track per match.

Unmatched detections create tentative tracks. Repeated support confirms them;
temporary misses produce coasting tracks; excessive misses delete them. Track
IDs are generated locally. Detection and tracking APIs never receive simulation
target IDs or exact states. Ground truth is used only for signal synthesis,
optional display, and benchmark evaluation.

### Track uncertainty on the PPI

For the track's upper-left position covariance block $P_{xy}$, the GUI computes
eigenvectors $Q$ and nonnegative eigenvalues $\lambda_1,\lambda_2$. Its 95%
two-dimensional Gaussian contour is

$$
\mathbf{p}(\phi)=\hat{\mathbf{p}}+
Q\begin{bmatrix}\sqrt{\lambda_1}&0\\0&\sqrt{\lambda_2}\end{bmatrix}
\sqrt{-2\ln(1-0.95)}
\begin{bmatrix}\sin\phi\\\cos\phi\end{bmatrix}.
$$

Velocity arrows start at the track estimate and end at the position predicted
by five seconds of constant-velocity motion. They are visualization aids, not
new tracker measurements. Tentative, confirmed, and coasting tracks have
distinct markers/colors. Hidden truth is never used to construct these overlays.
