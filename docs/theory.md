Signal Processing and Tracking Theory

Echorin is an educational radar and sonar simulator. Its algorithms are real,
but its point-target propagation and ideal angular channels are deliberately
simpler than high-fidelity electromagnetic or acoustic models.

Geometry and two-way propagation

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

Waveforms and matched filtering

The rectangular pulse is $s(t)=A\cos(2\pi f_ct)$. The LFM waveform is

$$
s(t)=A\cos\left(2\pi(f_0t+\tfrac12kt^2)\right),\qquad k=B/T_p.
$$

The matched filter correlates received samples with the conjugate time-reversed
transmit waveform. Echorin retains nonnegative lags and aligns output index zero
with zero delay. Pulse compression therefore produces a range peak without
looking at target truth.

CA-CFAR detection

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

Doppler processing

A coherent pulse interval samples target phase at the PRF. Echorin applies an
optional Hann window and FFT along slow time for every range bin. For wavelength
$\lambda=c/f_c$, monostatic Doppler and radial velocity obey

$$
f_d=\frac{2v_r}{\lambda},\qquad v_r=\frac{f_d\lambda}{2}.
$$

The velocity resolution is $\Delta v=(\mathrm{PRF}/N_p)\lambda/2$, and the
unambiguous interval follows $|f_d|<\mathrm{PRF}/2$. Echorin rejects simulated
targets outside that interval instead of silently aliasing them.

Kalman tracking and association

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