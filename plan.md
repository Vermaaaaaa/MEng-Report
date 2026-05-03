# Thesis Plan — Deep Learning Aided Sensor Fusion for Drift-Reduced IMU Orientation Estimation

🟢 Fully covered in good detail in interim report
🟡 Covered but needs more depth
🔴 Not present in interim report

---

> **Ordering note — two changes from original:**
> **Datasets moved before Deep Learning Architecture** because the architecture decisions (RF constraint, 9-channel input, noise augmentation) are directly motivated by properties of the BROAD dataset. Introducing the data first lets the architecture section justify its choices by referring back to it.
> **"Unaided Sensor Fusion" renamed to "Unaided Kalman Filter Baseline"** to make clear this is a comparative benchmark, which frames the deep learning chapters that follow as a motivated improvement.

---

## 1. Abstract
🟡 One paragraph (~200–250 words) — problem, approach, dataset, headline result. Semester 1 abstract present but scoped only to interim work; needs full rewrite once results are in. Write this **last**.

---

## 2. Acknowledgements
🟢 Supervisor Prof. Zhiqiang Zhang, BROAD and RepoIMU dataset authors. Present and appropriate — minor updates only.

---

## 3. Introduction

### 3.1 IMUs and their applications
🟢 What an IMU is (gyroscope, accelerometer, magnetometer), key domains (BSNs, robotics, autonomous vehicles), why self-contained high-rate sensing is attractive.

### 3.2 Problem statement — gyroscopic drift and its impact
🟢 Drift from angular rate integration, measured angular rate model ω_m = ω + b + n, how bias and noise propagate into quaternion update errors, Kianifar et al. on yaw limitation, lower-grade IMU severity.

### 3.3 Research questions and hypotheses
🟢 Primary question (can DL learn drift patterns for drift-free orientation?), sub-questions (generalisation, vs UKF, accuracy/cost trade-off), hypothesis (measurable reduction in geodesic error).

### 3.4 Report structure overview
🔴 One short paragraph signposting what each chapter covers. Absent — needs to be added for the final report.

---

## 4. IMU Basics and Operational Principles

### 4.1 Gyroscope — operation and errors
🟢 Angular rate measurement, quaternion representation (no gimbal lock, computationally efficient), integration kinematic update, bias and noise accumulation. Error sources: constant bias (degrees/h, linear drift), ARW/white noise (degrees/sqrt(h), sqrt(t) growth).

### 4.2 Accelerometer — operation and errors
🟢 Specific force measurement, gravity reference for roll/pitch, cannot correct yaw, only reliable under low linear acceleration. Error sources: constant offset bias, high linear acceleration as dominant dynamic failure mode.

### 4.3 Magnetometer — operation and errors
🟢 Local magnetic field, heading (yaw) reference, required for full 3-DOF orientation. Error sources: hard-iron (constant additive field), soft-iron (field-direction-dependent distortion), external magnetic disturbances.

---

## 5. Dataset (BROAD)

### 5.1 BROAD dataset
🔴 Full name, why selected (isolated disturbances, rest phases, short/long motions), sensors, sampling rate (286 Hz), optical motion capture ground truth (quaternions), train/val/test split. Mentioned briefly in interim (sections 3.2 and abstract) but no dedicated section.

### 5.2 Scope and limitations
🔴 Single dataset, controlled lab setting, potential domain shift to unconstrained real-world motion, generalisation intent (RepoIMU). Not discussed anywhere in interim.

---

## 6. Unaided Kalman Filter Baseline

### 6.1 UKF overview
🔴 Why UKF over KF (non-linear quaternion kinematics) and over EKF (empirically better, cite Kraft 2003). State vector: quaternion (4) + gyro bias (3). Process model: gyro integration with slowly varying bias. Measurement model: accelerometer as gravity reference, magnetometer as Earth-field reference.

### 6.2 Filter configuration
🔴 Q and R estimated from BROAD rest phases, adaptive gating (scale R to downweight unreliable measurements rather than outright rejection which causes discontinuities).

### 6.3 Scenarios and error metric
🔴 Geodesic angle error metric, sign convention (q = -q), three BROAD scenarios: undisturbed slow rotation, undisturbed fast rotation, disturbed stationary magnet.

### 6.4 Results
🔴 All three scenarios with tables (mean, std, median, min, max). Slow rotation: mean ~5.8 deg, brief spikes, good recovery. Fast rotation: mean ~11.5 deg, max ~73 deg, accelerometer corruption. Disturbed magnet: mean ~37.8 deg, max ~180 deg, adaptive gating insufficient.

### 6.5 Discussion
Conclude: UKF sets clear performance ceiling deep learning must improve upon. Suggest that errors in IMU section is what causes the limitations for thr UKF and why DL can be used to help this. 

---

## 7. Deep Learning Architecture

### 7.1 Aims of the model
🟡 Predict per-timestep correction delta_omega_t; corrected rate integrated via SO3 exp map into orientation increments; all three IMU modalities allow context-dependent corrections. Briefly stated in interim section 4.1 — needs expanding with full integration rationale.

### 7.2 Model selection

#### 7.2.1 Recurrent Neural Networks
🟢 Recurrence equation, BPTT, vanishing/exploding gradients, no parallelism leading to slow training, LSTM/GRU mitigate but do not solve.

#### 7.2.2 Convolutional Neural Networks
🟢 Local feature extraction via learnable filter, parallelisable training, stable gradients, but cannot model long-range temporal dependencies.

#### 7.2.3 Temporal Convolutional Networks
🟢 Causal dilated convolutions (no data leakage), dilation grows exponentially per layer, combines CNN efficiency with RNN-like context. Cite Bai et al.

#### 7.2.4 Justification for RF constraint (>=286, <=1024 samples)
🟡 >=286 = one full second at 286 Hz (timescale of visible bias accumulation). <=1024 = computational budget and diminishing returns. Mentioned as "approximately two seconds" in interim section 5.1.1 — quantitative upper bound rationale absent.

### 7.3 TCN architecture design

#### 7.3.1 Residual block structure
🟢 Two dilated causal convolutions per block, each followed by normalisation, activation, dropout. Residual connection with 1x1 convolution (channel projection) if in/out channels differ. Reproduces interim Figure 10.

#### 7.3.2 Normalisation
🟡 Options: BatchNorm, LayerNorm (GroupNorm with 1 group), WeightNorm, combinations. Table present in interim. Final chosen method (from Phase 2: batchnorm_only or batchnorm_weight_norm) not yet justified — awaits Phase 2 results.

#### 7.3.3 Activation functions
🟡 Options: ReLU, LeakyReLU (alpha=0.1), GELU, Swish/SiLU, Mish. Table present in interim. Final choice (LeakyReLU from Phase 1) not yet documented in report.

#### 7.3.4 Regularisation
🟡 Dropout after each activation (range 0.0–0.4) mentioned in interim. Weight decay (AdamW) and gradient clipping (max_norm=2.0) from training loop are absent — need adding.

#### 7.3.5 Final architecture summary
🔴 Winning channel configuration, kernel size, dilation, computed RF, total parameter count. Correctly listed as semester 2 aim in interim — must be filled from Phase 2 Optuna best trial.

---

## Data preprocessing piepline
### 5.2 Data pre-processing
🔴 Normalisation (per-axis mean/std from training set, online computation, saved and reused), windowing strategy (random start for training, fixed non-overlapping windows for val/test, 2018 samples ~7.06 s), noise augmentation (additive Gaussian from BROAD rest-phase measurements, applied post-normalisation), input tensor format [B, 9, T] and target format [B, T, 4]. Entirely absent — datasets.py implements all of this but it is undocumented in the report.


## 8. Loss Function Design

### 8.1 Motivation for SO3-aware loss
🔴 Why not MSE on raw quaternion components: unit-norm constraint hard to enforce, sign ambiguity (q = -q) causes instability, MSE in quaternion space does not equal angular distance. Instead predict angular rate corrections and integrate via SO3. Entirely absent — critical section and core novelty of the implementation.

### 8.2 SO3 exponential map
🔴 Maps predicted rotation vector omega in R^3 to unit quaternion delta_q. Linear approximation for small theta to avoid division by zero. Code: SO3_exp_map in lie_algebra.py. Implemented but undocumented in report.

### 8.3 Pairwise quaternion accumulation
🔴 Sequential composition gives full orientation trajectory. Naive O(T) approach replaced with parallel pairwise tree reduction achieving O(log T). Re-normalise after accumulation. Code: pairwise_reduction in losses.py. One of the more technically interesting parts of the implementation — entirely absent.

### 8.4 Multi-scale supervised loss
🔴 Relative quaternion ground truth at decimation points spaced base_interval * 2^k samples apart (base_interval=16). Sign alignment before error computation. SO3 log map to get rotation vector; Huber loss vs zero vector. Scale weighting 1/2^k. receptive_field parameter masks warm-up region. Entirely absent.

### 8.5 Geodesic evaluation metric
🟡 Geodesic angle error, evaluated at finest scale only, used as Optuna objective. Defined in interim section 3.4.2 under UKF evaluation — should be referenced/re-introduced here as the primary DL metric.

---

## 9. Optimisations (Hyperparameter Search)

### 9.1 Phase 1 — architecture search
🔴 Search space: n_layers (2–7), base_channels (8–32), channel growth (double/constant/gradual), kernel_size (3–9), dilation (2–4), norm_type (5 options), activation (5 options), dropout (0.05–0.4), lr, weight_decay, scheduler. RF constraint [286, 1024] as hard prune. TPE sampler (multivariate, 100 startup trials). Hyperband pruner (min=50, max=150, reduction=3). 150 epochs/trial, batch 192, 750 trials. Listed as semester 2 aim in interim — entirely absent.

### 9.2 Phase 2 — regularisation fine-tuning
🔴 Architecture fixed from Phase 1. Narrowed search: norm_type (batchnorm_only vs batchnorm_weight_norm), dropout (0.0–0.4), lr, weight_decay. Scheduler fixed to OneCycleLR. MedianPruner (n_startup=40, warmup=50, interval=10). 100 epochs, 200 trials. Absent.

### 9.3 Training setup
🔴 AdamW optimiser, bfloat16 mixed-precision autocast, gradient clipping max_norm=2.0, OOM catching with trial pruning and GPU memory release. Absent.

### 9.4 Final model selection
🔴 Winning trial hyperparameters, validation geodesic loss, receptive field, parameter count. Awaits completion of Phase 2 runs.

---

## 10. Results

### 10.1 Experimental setup
🔴 Hardware (GPU, training time), test set description (files unseen during training/validation), metrics reported.

### 10.2 Comparison against UKF baseline
🔴 Same three BROAD scenarios as Chapter 6 for direct comparison. Error tables (mean, std, median, min, max). Percentage improvement in mean and median geodesic error.

### 10.4 Ablation studies (time permitting)
🔴 Effect of noise augmentation (with vs. without), multi-scale vs. single-scale loss, RF size sweep.

---

## 11. Discussion

### 11.1 Interpretation of results
🔴 Where model improved over UKF and why (e.g. during magnetic disturbances learns to rely on gyro/acc cues). Where it fell short. Whether drift accumulates slowly or is bounded over time.

### 11.2 Limitations
🔴 Single dataset (controlled lab setting, unproven in real-world motion), causal constraint (no future context, offline smoothing could do better but not real-time), latency vs KF for embedded use, TCN domain transfer at different sampling rates.

### 11.3 Societal and ethical factors
🔴 Healthcare / BSNs (accurate orientation from low-cost IMUs enables clinical gait analysis, rehabilitation, prosthetics). Privacy (wearable IMU data can infer location, activity, health status). Reliability and safety (overconfident estimates from a poorly generalised model may be more dangerous than a known-bad filter estimate — no uncertainty quantification present). Environmental (MEMS IMUs lower footprint than GPS; improving accuracy supports GPS-denied use cases). **Worth 10% of overall mark per rubric — needs substantive treatment, not a single paragraph.**

---

## 12. Conclusion

### 12.1 Summary of contributions
🟡 UKF baseline established, TCN with SO3-aware multi-scale loss designed, two-phase Optuna search conducted, geodesic error improvement vs UKF demonstrated. Semester 1 conclusion exists but covers only interim progress — full rewrite needed around final results.

### 12.2 Future work
🟡 Integration of correction into UKF as learned process noise, testing on additional datasets, quantisation/distillation for embedded deployment, probabilistic outputs for uncertainty-aware fusion. Semester 2 aims section (5.1) covers some of this — needs retrospective expansion once work is done.

---

## Recommended Writing Order
vv
Write sections in this order to avoid blocking yourself on missing results:

1. **Datasets (5)** — you know the code cold; fast to write
2. **KF Baseline (6)** — mostly done in interim; expand and tidy
3. **IMU Basics (4)** — background clear; expand from interim
4. **Loss Function (8)** — code is the source of truth; translate to maths and prose
5. **Architecture (7)** — reference dataset and loss sections already written
6. **Optimisations (9)** — document search methodology and winning parameters
7. **Results (10)** — fill once experiments complete
8. **Discussion (11)** — write after results
9. **Introduction (3)** — easier once you know the full story
10. **Conclusion (12)** — mirrors introduction; write second-to-last
11. **Abstract (1)** — always last