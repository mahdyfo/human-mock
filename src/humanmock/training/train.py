"""Trains the body against data.

Training runs in two stages, because the order matters.

Calibration only adjusts physiological constants and measurement noise. It finds
the best version of the known equations, so the networks do not learn to
compensate for a wrong constant.

Neural training then also opens the organ cells, the memory neurons, the patient
encoder, and the observation network. They can only add what the equations miss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from humanmock.checkpoint import save_checkpoint
from humanmock.data.formats import Record, assign_splits, load_records, sample_people, support_summary
from humanmock.knowledge import DRUG_FIELDS, KNOWLEDGE, MAX_DRUGS
from humanmock.model import OBS_NAMES, TRAIN_SOLVER, BodyModel, SolverConfig, advance

_TABLE = jnp.asarray(KNOWLEDGE.drug_table)
MONTE_CARLO_VARIANCE = 0.03**2


@dataclass(frozen=True)
class TrainConfig:
    calibration_steps: int = 400
    neural_steps: int = 800
    batch_size: int = 8
    people: int = 8
    learning_rate: float = 0.02
    neural_learning_rate: float = 0.002
    prior_strength: float = 0.02
    weight_decay: float = 1e-5
    eval_every: int = 50
    patience: int = 8
    seed: int = 0
    val_fraction: float = 0.2


def _forward_person(model, cov, es, ep, ed, b, cfg: SolverConfig):
    """Simulates one virtual person along a subject's timeline and returns every observable."""
    c = model.context(cov, es, ep, ed, _TABLE)
    y = model.initial_state(c)
    zero = jnp.zeros(MAX_DRUGS)
    slot = b["path_slot"]
    onehot = jax.nn.one_hot(jnp.maximum(slot, 0), KNOWLEDGE.n_pathogens)
    inoculum = c.path["log10_inoculum"]
    exposure0 = jnp.where((slot >= 0) & (onehot > 0), inoculum, -10.0)
    y = model.apply_events(y, c, zero, zero, exposure0)
    burn = jnp.where(slot >= 0, c.path["incubation_h"][jnp.maximum(slot, 0)] + b["onset_days"] * 24.0, 0.0)
    y = advance(model, y, c, burn, cfg)

    def step(y_prev, seg):
        dt, oral, iv, exposure = seg
        y_now = advance(model, y_prev, c, dt, cfg)
        obs = model.observe(y_now, c)
        return model.apply_events(y_now, c, oral, iv, exposure), obs

    _, trajectory = jax.lax.scan(step, y, (b["dt"], b["oral"], b["iv"], b["exposure"]))
    return trajectory


def predict_subject(model, b, key, cfg: SolverConfig):
    """The expected measurement for one subject, averaged across their virtual people."""
    people = b["cov"].shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    es = jax.random.normal(k1, (people, len(KNOWLEDGE.sys_names)))
    ep = jax.random.normal(k2, (people, KNOWLEDGE.n_pathogens, len(KNOWLEDGE.path_fields)))
    ed = jax.random.normal(k3, (people, MAX_DRUGS, len(DRUG_FIELDS)))
    traj = jax.vmap(lambda cov, a, c, d: _forward_person(model, cov, a, c, d, b, cfg))(b["cov"], es, ep, ed)
    good = jnp.all(jnp.isfinite(traj), axis=(1, 2))
    traj = jnp.where(good[:, None, None], traj, 0.0)
    mean = jnp.sum(traj, axis=0) / jnp.maximum(jnp.sum(good), 1)
    return mean[b["obs_index"], b["obs_var"]], jnp.mean(good)


def _subject_nll(model, b, key, cfg):
    pred, ok = predict_subject(model, b, key, cfg)
    noise = jnp.exp(model.log_obs_sigma)[b["obs_var"]]
    # A reported arm mean carries its own uncertainty, and an individual reading also has measurement noise.
    # The small floor covers sampling noise from simulating a limited number of virtual people.
    variance = b["obs_sd"] ** 2 + jnp.where(b["is_arm"], 0.0, noise**2) + MONTE_CARLO_VARIANCE
    residual = (pred - b["obs_value"]) ** 2 / variance
    nll = 0.5 * residual + 0.5 * jnp.log(variance)
    return jnp.sum(nll * b["weight"] * b["mask"]), jnp.sum(b["weight"] * b["mask"]), ok


def _penalty(model, config: TrainConfig):
    prior = jnp.sum(model.log_sys**2) + jnp.sum(model.log_path**2) + jnp.sum(model.log_drug**2)
    return config.prior_strength * prior


@eqx.filter_jit
def _loss_batch(model, batch, key, config: TrainConfig, cfg: SolverConfig):
    keys = jax.random.split(key, batch["dt"].shape[0])
    total, weight, ok = jax.vmap(lambda b, k: _subject_nll(model, b, k, cfg))(batch, keys)
    data = jnp.sum(total) / jnp.maximum(jnp.sum(weight), 1.0)
    return data + _penalty(model, config) / jnp.maximum(jnp.sum(weight), 1.0), (data, jnp.mean(ok))


@eqx.filter_jit
def _train_step(diff, static, opt_state, batch, key, optimizer, config, cfg):
    def loss(d):
        return _loss_batch(eqx.combine(d, static), batch, key, config, cfg)

    (value, aux), grads = eqx.filter_value_and_grad(loss, has_aux=True)(diff)
    updates, opt_state = optimizer.update(grads, opt_state, diff)
    return eqx.apply_updates(diff, updates), opt_state, value, aux


class Batcher:
    """Groups subjects with similar timelines, so each group compiles only once."""

    def __init__(self, records: list[Record], batch_size: int, seed: int):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        counts: dict[int, int] = {}
        for record in records:
            for v in record.obs_var:
                counts[int(v)] = counts.get(int(v), 0) + 1
        # Each kind of measurement counts about equally, so plentiful ones cannot drown rare ones.
        self.var_weight = {v: 1.0 / np.sqrt(c) for v, c in counts.items()}
        mean_weight = np.mean(list(self.var_weight.values())) if self.var_weight else 1.0
        self.var_weight = {v: w / mean_weight for v, w in self.var_weight.items()}
        self.buckets: dict[tuple[int, int], list[Record]] = {}
        for record in records:
            self.buckets.setdefault(self._key(record), []).append(record)
        self.keys = sorted(self.buckets)
        sizes = np.array([len(self.buckets[k]) for k in self.keys], dtype=float)
        self.probabilities = sizes / sizes.sum() if len(sizes) else sizes

    @staticmethod
    def _key(record: Record) -> tuple[int, int]:
        def round_up(n: int) -> int:
            return int(2 ** np.ceil(np.log2(max(n, 8))))

        return round_up(len(record.dt)), round_up(len(record.obs_value))

    def collate(self, records: list[Record], T: int, K: int) -> dict[str, Any]:
        def pad(array, length, value=0.0):
            shape = (length - array.shape[0],) + array.shape[1:]
            return np.concatenate([array, np.full(shape, value, dtype=array.dtype)], axis=0)

        out = {
            "cov": np.stack([r.cov for r in records]),
            "path_slot": np.array([r.path_slot for r in records], dtype=np.int32),
            "onset_days": np.array([r.onset_days for r in records], dtype=np.float32),
            "is_arm": np.array([r.is_arm for r in records], dtype=bool),
            "dt": np.stack([pad(r.dt, T) for r in records]),
            "oral": np.stack([pad(r.oral, T) for r in records]),
            "iv": np.stack([pad(r.iv, T) for r in records]),
            "exposure": np.stack([pad(r.exposure, T, -10.0) for r in records]),
            "obs_index": np.stack([pad(r.obs_index, K) for r in records]),
            "obs_var": np.stack([pad(r.obs_var, K) for r in records]),
            "obs_value": np.stack([pad(r.obs_value, K) for r in records]),
            "obs_sd": np.stack([pad(r.obs_sd, K) for r in records]),
            "mask": np.stack([pad(np.ones(len(r.obs_value), dtype=np.float32), K) for r in records]),
            "weight": np.stack(
                [pad(np.array([self.var_weight.get(int(v), 1.0) for v in r.obs_var], dtype=np.float32), K) for r in records]
            ),
        }
        return {k: jnp.asarray(v) for k, v in out.items()}

    def _size(self, pool: list[Record]) -> int:
        # A small group never repeats a subject just to fill a batch.
        return min(self.batch_size, len(pool))

    def sample(self) -> dict[str, Any]:
        key = self.keys[int(self.rng.choice(len(self.keys), p=self.probabilities))]
        pool = self.buckets[key]
        chosen = self.rng.choice(len(pool), self._size(pool), replace=False)
        return self.collate([pool[i] for i in chosen], *key)

    def all_batches(self):
        """Every subject once, for evaluation. Short groups repeat with zero weight."""
        for key in self.keys:
            pool = self.buckets[key]
            size = self._size(pool)
            for start in range(0, len(pool), size):
                chunk = pool[start : start + size]
                filler = chunk + [chunk[-1]] * (size - len(chunk))
                batch = self.collate(filler, *key)
                if len(chunk) < size:
                    m = np.array(batch["mask"])
                    m[len(chunk) :] = 0.0
                    batch["mask"] = jnp.asarray(m)
                yield batch


def evaluate_loss(model, records: list[Record], batch_size: int, key, cfg: SolverConfig) -> float:
    if not records:
        return float("nan")
    batcher = Batcher(records, batch_size, 0)
    config = TrainConfig(prior_strength=0.0)
    total, weight = 0.0, 0.0
    for i, batch in enumerate(batcher.all_batches()):
        (_, (data, _)) = _loss_batch(model, batch, jax.random.fold_in(key, i), config, cfg)
        w = float(jnp.sum(batch["mask"] * batch["weight"]))
        total += float(data) * w
        weight += w
    return total / max(weight, 1e-9)


def _readable(observable: str) -> str:
    """Names a drug slot by its drug, matching the names used in the data files."""
    if observable.startswith("conc_slot"):
        slot = int(observable[len("conc_slot") :])
        name = KNOWLEDGE.drug_names[slot] if slot < len(KNOWLEDGE.drug_names) else f"slot{slot}"
        return f"conc_{name}_mg_l"
    return observable


@eqx.filter_jit
def _predict_batch(model, batch, keys, cfg: SolverConfig):
    return jax.vmap(lambda b, k: predict_subject(model, b, k, cfg))(batch, keys)


def per_variable_error(model, records: list[Record], batch_size: int, key, cfg: SolverConfig) -> dict[str, dict[str, float]]:
    """Plain-language score: typical error per measurement, in the measurement's own units."""
    batcher = Batcher(records, batch_size, 0)
    errors: dict[int, list[float]] = {}
    for i, batch in enumerate(batcher.all_batches()):
        keys = jax.random.split(jax.random.fold_in(key, i), batch["dt"].shape[0])
        pred, _ = _predict_batch(model, batch, keys, cfg)
        pred = np.asarray(pred)
        value, var, mask = (np.asarray(batch[k]) for k in ("obs_value", "obs_var", "mask"))
        for a, b, v, m in zip(pred.ravel(), value.ravel(), var.ravel(), mask.ravel()):
            if m > 0:
                errors.setdefault(int(v), []).append(float(a - b))
    return {
        _readable(OBS_NAMES[v]): {"rmse": float(np.sqrt(np.mean(np.square(e)))), "n": len(e)}
        for v, e in sorted(errors.items())
    }


def _run_stage(
    model: BodyModel,
    stage: str,
    steps: int,
    lr: float,
    train: list[Record],
    val: list[Record],
    config: TrainConfig,
    cfg: SolverConfig,
    key,
    log: Callable[[str], None],
) -> tuple[BodyModel, list[dict[str, float]]]:
    if steps <= 0 or not train:
        return model, []
    spec = _stage_filter(model, stage)
    diff, static = eqx.partition(model, spec)
    schedule = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(5.0), optax.adamw(schedule, weight_decay=config.weight_decay)),
        max_consecutive_errors=25,
    )
    opt_state = optimizer.init(diff)
    batcher = Batcher(train, config.batch_size, config.seed)

    best = eqx.combine(diff, static)
    eval_key = jax.random.PRNGKey(config.seed + 1234)
    best_score = evaluate_loss(best, val if val else train, config.batch_size, eval_key, cfg)
    history = [{"stage": stage, "step": 0, "val": best_score}]
    log(f"  {stage}: starting validation score {best_score:.4f}")
    stale = 0
    started = time.time()
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        diff, opt_state, value, (data, ok) = _train_step(
            diff, static, opt_state, batcher.sample(), sub, optimizer, config, cfg
        )
        if step % config.eval_every == 0 or step == steps:
            current = eqx.combine(diff, static)
            score = evaluate_loss(current, val if val else train, config.batch_size, eval_key, cfg)
            improved = np.isfinite(score) and score < best_score - 1e-4
            history.append({"stage": stage, "step": step, "train": float(data), "val": score})
            log(
                f"  {stage} step {step}/{steps}: train {float(data):.4f}, validation {score:.4f}"
                f"{' *' if improved else ''}, {time.time() - started:.0f} s"
            )
            if improved:
                best, best_score, stale = current, score, 0
            else:
                stale += 1
                if stale >= config.patience:
                    log(f"  {stage}: stopping early, validation stopped improving.")
                    break
    return best, history


def _stage_filter(model: BodyModel, stage: str):
    """True for parameters this stage may change."""
    arrays = eqx.is_inexact_array
    if stage == "neural":
        return jax.tree_util.tree_map(arrays, model)
    spec = jax.tree_util.tree_map(lambda _: False, model)
    return eqx.tree_at(
        lambda m: (m.log_sys, m.log_path, m.log_drug, m.log_iiv, m.log_obs_sigma),
        spec,
        (True, True, True, True, True),
    )


def train_member(
    train: list[Record],
    val: list[Record],
    config: TrainConfig,
    log: Callable[[str], None] = print,
    cfg: SolverConfig = TRAIN_SOLVER,
) -> tuple[BodyModel, list[dict[str, float]]]:
    key = jax.random.PRNGKey(config.seed)
    k_init, k_cal, k_neural = jax.random.split(key, 3)
    model = BodyModel.create(k_init)
    model, h1 = _run_stage(model, "calibration", config.calibration_steps, config.learning_rate, train, val, config, cfg, k_cal, log)
    model, h2 = _run_stage(model, "neural", config.neural_steps, config.neural_learning_rate, train, val, config, cfg, k_neural, log)
    return model, h1 + h2


def run_training(
    data_root: Path,
    out_dir: Path,
    config: TrainConfig,
    members: int = 1,
    include_synthetic: bool = False,
    only_sources: list[str] | None = None,
    holdout_studies: list[str] | None = None,
    holdout_drugs: list[str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    records = load_records(data_root, include_synthetic=include_synthetic, only=only_sources)
    if not records:
        raise SystemExit(f"No training data found under {data_root}. Run the data commands first.")
    sample_people(records, config.people, config.seed)
    assign_splits(records, config.seed, config.val_fraction, holdout_studies, holdout_drugs)
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    holdout = [r for r in records if r.split == "holdout"]
    log(f"Subjects: {len(train)} training, {len(val)} validation, {len(holdout)} held out")
    log(f"Studies in training: {sorted({r.study_id for r in train})}")
    log(f"Studies in validation: {sorted({r.study_id for r in val})}")
    if holdout:
        log(f"Studies held out: {sorted({r.study_id for r in holdout})}")

    trained, histories = [], []
    for m in range(members):
        log(f"Training ensemble member {m + 1} of {members}")
        member_config = TrainConfig(**{**config.__dict__, "seed": config.seed + m})
        model, history = train_member(train, val, member_config, log)
        trained.append(model)
        histories.append(history)

    key = jax.random.PRNGKey(4242)
    prior = BodyModel.create(jax.random.PRNGKey(0))
    scores: dict[str, Any] = {}
    for name, part in (("train", train), ("validation", val), ("holdout", holdout)):
        if part:
            scores[name] = {
                "untrained": per_variable_error(prior, part, config.batch_size, key, TRAIN_SOLVER),
                "trained": per_variable_error(trained[0], part, config.batch_size, key, TRAIN_SOLVER),
            }
    meta = {
        "support": support_summary(records),
        "data": {
            "sources": sorted({r.source for r in records}),
            "train_studies": sorted({r.study_id for r in train}),
            "validation_studies": sorted({r.study_id for r in val}),
            "holdout_studies": sorted({r.study_id for r in holdout}),
            "synthetic_included": any(r.source == "synthetic" for r in records),
        },
        "training": {k: v for k, v in config.__dict__.items()},
        "scores": scores,
        "history": histories,
    }
    save_checkpoint(out_dir, trained, meta)
    log(f"Saved to {out_dir}")
    return meta
