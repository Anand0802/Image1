# ── 1. Config additions ──────────────────────────────────
CFG.LAMBDA_PHASE   = 0.3
CFG.PHASE_PROJ_DIM = 64

# ── 2. Phase extraction utility ──────────────────────────
from scipy.signal import hilbert as sp_hilbert

def extract_instantaneous_phase(ppg):
    ppg_f = bandpass(ppg, 0.5, 8.0, CFG.FS_PPG_TARGET)
    phase = np.unwrap(np.angle(sp_hilbert(ppg_f.astype(np.float64))))
    return zscore(phase.astype(np.float32))

# ── 3. Phase encoder + projection head ───────────────────
def build_phase_encoder(win_len, d=CFG.D_MODEL):
    inp = layers.Input(shape=(win_len, 1), name="phase_in")
    x = layers.Conv1D(32, 7, padding="same", kernel_initializer="he_normal")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Conv1D(64, 5, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = se_block(x, ratio=4)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(d, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(d, activation="relu")(x)
    x = layers.LayerNormalization(name="phase_emb")(x)
    return models.Model(inp, x, name="PhaseEncoder")

def build_phase_proj_head(d=CFG.D_MODEL, proj=CFG.PHASE_PROJ_DIM):
    inp = layers.Input((d,))
    x = layers.Dense(d, activation="relu")(inp)
    x = layers.Dense(proj)(x)
    x = layers.LayerNormalization(name="phase_proj_out")(x)
    return models.Model(inp, x, name="PhaseProjHead")

# ── 4. Patch SSLTrainer with phase support
import types

def _new_init(self, encoder, proj_head, augmentor,
              beat_proj_head=None, phase_encoder=None, phase_proj_head=None):
    self.enc = encoder; self.proj = proj_head; self.aug = augmentor
    self.beat_proj = beat_proj_head; self.phase_enc = phase_encoder
    self.phase_proj = phase_proj_head
    self.α_beat = CFG.ALPHA_BEAT; self.λ_phase = CFG.LAMBDA_PHASE
    self.beat_det = _BeatDetectorLight(fs=CFG.FS_PPG_TARGET)
    self.cnn_feat_model = SSLTrainer._build_cnn_feat_model(encoder) if beat_proj_head else None
    try:
        self.opt = optimizers.AdamW(CFG.SSL_LR, weight_decay=1e-4)
        self.opt_phase = optimizers.AdamW(CFG.SSL_LR, weight_decay=1e-4)
    except AttributeError:
        self.opt = optimizers.Adam(CFG.SSL_LR)
        self.opt_phase = optimizers.Adam(CFG.SSL_LR)
    self.backbone_vars = self.enc.trainable_variables + self.proj.trainable_variables

def _phase_contrast_step(self, ppg_batch):
    B = len(ppg_batch)
    pv1 = tf.constant(np.stack([extract_instantaneous_phase(
              self.aug.augment(ppg_batch[i])) for i in range(B)])[:, :, None], dtype=tf.float32)
    pv2 = tf.constant(np.stack([extract_instantaneous_phase(
              self.aug.augment(ppg_batch[i])) for i in range(B)])[:, :, None], dtype=tf.float32)
    phase_vars = self.phase_enc.trainable_variables + self.phase_proj.trainable_variables
    with tf.GradientTape() as tape:
        l_phase = nt_xent_loss(self.phase_proj(self.phase_enc(pv1, training=True), training=True),
                               self.phase_proj(self.phase_enc(pv2, training=True), training=True))
    self.opt_phase.apply_gradients(zip(tape.gradient(l_phase, phase_vars), phase_vars))
    return float(l_phase)

def _new_train(self, joint_ds, epochs=CFG.SSL_EPOCHS):
    has_hcl = self.beat_proj is not None
    has_phase = self.phase_enc is not None
    print(f"\n{'═'*60}\n  SSL PRE-TRAINING ({epochs} epochs)\n{'═'*60}")
    _obj = "L_segment" + (f" + {self.α_beat}×L_beat" if has_hcl else "") \
                        + (f" + {self.λ_phase}×L_phase" if has_phase else "")
    print(f"  Objective : {_obj}\n")
    best_loss = float("inf")
    ckpt = os.path.join(CFG.SAVE_DIR, "models", "ssl_encoder_best.h5")
    pckpt = os.path.join(CFG.SAVE_DIR, "models", "ssl_phase_enc_best.h5")
    epoch_logs = []
    for ep in range(1, epochs + 1):
        seg_l, beat_l, phase_l = [], [], []
        for ppg_b, _ in joint_ds.balanced_batches(CFG.SSL_BATCH):
            v1, v2 = self._prepare_batch(ppg_b)
            seg_l.append(float(self._train_step(tf.constant(v1), tf.constant(v2))))
            if has_hcl:
                bp = self._extract_beat_positions(ppg_b)
                if sum(len(p) for p in bp) >= 2:
                    beat_l.append(self._beat_contrast_step(v1, v2, bp))
            if has_phase:
                phase_l.append(self._phase_contrast_step(ppg_b))
        ep_seg = float(np.mean(seg_l))
        ep_beat = float(np.mean(beat_l)) if beat_l else 0.0
        ep_phase = float(np.mean(phase_l)) if phase_l else 0.0
        ep_total = ep_seg + self.α_beat * ep_beat + self.λ_phase * ep_phase
        epoch_logs.append({"epoch": ep, "loss": ep_total, "l_segment": ep_seg,
                            "l_beat": ep_beat, "l_phase": ep_phase,
                            "beat_steps": len(beat_l), "phase_steps": len(phase_l)})
        if ep_total < best_loss:
            best_loss = ep_total; self.enc.save_weights(ckpt)
            if has_phase: self.phase_enc.save_weights(pckpt)
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{epochs}  L_total={ep_total:.4f}  "
                  f"L_seg={ep_seg:.4f}" +
                  (f"  L_beat={ep_beat:.4f}" if has_hcl else "") +
                  (f"  L_phase={ep_phase:.4f}" if has_phase else "") +
                  f"  best={best_loss:.4f}")
    self.enc.load_weights(ckpt)
    if has_phase and os.path.exists(pckpt): self.phase_enc.load_weights(pckpt)
    print(f"\n  ✓ SSL complete. Best loss: {best_loss:.4f}\n")
    log_path = os.path.join(CFG.SAVE_DIR, "logs", "ssl_pretrain_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
        w.writeheader(); w.writerows(epoch_logs)
    return epoch_logs

SSLTrainer.__init__          = _new_init
SSLTrainer._phase_contrast_step = _phase_contrast_step
SSLTrainer.train             = _new_train

# ── 5. Build & run ────────────────────────────────────────
win_len         = int(CFG.FS_PPG_TARGET * CFG.WIN_SEC)
encoder         = build_backbone(win_len)
proj_head       = build_proj_head()
beat_proj_head  = build_beat_proj_head(CFG.D_MODEL, CFG.BEAT_PROJ_DIM)
phase_encoder   = build_phase_encoder(win_len, CFG.D_MODEL)
phase_proj_head = build_phase_proj_head(CFG.D_MODEL, CFG.PHASE_PROJ_DIM)
augmentor       = PPGAugmentor(seed=CFG.SEED)

print(f"PhaseEncoder params : {phase_encoder.count_params():,}")

ssl_trainer = SSLTrainer(encoder, proj_head, augmentor,
                         beat_proj_head  = beat_proj_head,
                         phase_encoder   = phase_encoder,
                         phase_proj_head = phase_proj_head)
