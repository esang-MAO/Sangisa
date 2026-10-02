"""Export HT-Demucs to ONNX for the in-browser engine (site/engine/demucs.js).

The graph holds the network and the inverse spectrogram behind it, so it returns audio. The
spectrogram in front of it (a small input) and chunking the song stay in JavaScript:

    inputs   mix      float32 [1, 2, L]           one chunk of audio (L = the model's training length)
             spec     float32 [1, 4, 2048, T]     its spectrogram, complex as channels (demucs' "CaC")
    output   sources  float32 [1, S, 2, L]        the chunk split into S sources

Several changes keep it inside a phone browser's memory, all with the same result as PyTorch:
attention runs one head at a time, the shape arithmetic is folded at export (so the browser
needn't run its optimizer), and returning audio rather than per-source spectrograms saves a
176 MB output per chunk.

Usage:
    python scripts/export_demucs_onnx.py --model htdemucs --out site/models          # pretrained
    python scripts/export_demucs_onnx.py --tiny --out tests/engine/fixtures --reference  # test fixture

Writes <out>/<name>.onnx (the graph), <name>.weights.f16 or .f32 (the weights, which the graph
reads as the external data file <name>.onnx.data) and <name>.json (shapes, sources, file names).
With --reference also a short test signal and the PyTorch apply_model output, for parity tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Core(nn.Module):
    """HTDemucs.forward between the spectrogram and the masking/inverse spectrogram."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, mix, mag):
        m = self.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = m.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                # demucs expands this to x's shape first; broadcasting gives the same sum without
                # a 33 MB constant
                emb = m.freq_emb(frs).t()[None, :, :, None]
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_upsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = x.reshape(b, c, f * t)
                x = m.channel_downsampler(x)
                x = x.reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        L = mix.shape[-1]
        xt = xt.view(B, S, -1, L)
        xt = xt * stdt[:, None] + meant[:, None]
        return ispec(x, m.nfft, m.hop_length, L, zero=mix[0, 0, 0] * 0) + xt


def ispec(x, nfft: int, hop: int, length: int, zero=0.0):
    """HTDemucs._mask + _ispec (torch.istft) as plain tensor ops, so ONNX Runtime returns audio.

    x: [B, S, C*2, F, T] spectrograms, real and imaginary parts interleaved per channel; the
    Nyquist bin and demucs' two empty frames on each side are zero. Returns [B, S, C, length].
    The inverse DFT is two matrix products with a cosine and a sine table built from ranges
    inside the graph; frames overlap-add in four shifted slices. `zero` is 0 computed from
    the input: adding it keeps the tables from being folded into weights at export (which would
    add 33 MB to the download and round them to float16). ONNX Runtime builds them per chunk.
    """
    B, S, C2, Fq, T = x.shape
    N, half = nfft, nfft // 2
    assert Fq == half and N == 4 * hop
    re, im = x[:, :, 0::2].transpose(-1, -2), x[:, :, 1::2].transpose(-1, -2)  # [B, S, C, T, F]
    # k * n < 2**24, so this integer arithmetic is exact in float32 (ONNX Runtime Web has no
    # int64 -> float Cast); reducing mod N first keeps the angles precise.
    k = torch.arange(Fq, dtype=torch.float32) + zero
    n = torch.arange(half + 1, dtype=torch.float32) + zero
    phase = torch.remainder(k[:, None] * n[None, :], N)
    angle = phase * (2 * math.pi / N)
    # irfft with the DC bin counted once and the others twice, and istft's normalized=True scale
    weight = torch.where(k == 0, 1.0, 2.0)[:, None] * (math.sqrt(N) / N)
    a = re @ (torch.cos(angle) * weight)  # [B, S, C, T, half + 1]
    b = im @ (torch.sin(angle) * weight)
    # samples n and N - n share the same cosine and opposite sines
    frames = torch.cat([a - b, torch.flip((a + b)[..., 1:half], dims=[-1])], dim=-1)  # [.., T, N]
    i = torch.arange(N, dtype=torch.float32) + zero
    window = 0.5 - 0.5 * torch.cos(i * (2 * math.pi / N))  # periodic hann
    frames = (frames * window).reshape(B, S, C2 // 2, T, 4, hop)
    # overlap-add: block r of the output gets frame r - j's j-th quarter
    ola = sum(nn.functional.pad(frames[..., j, :], (0, 0, j, 3 - j)) for j in range(4))  # [.., T + 3, hop]
    # istft's window-square envelope over T + 4 frames (demucs pads two empty frames each side)
    sq = (window * window).reshape(4, hop)
    env = sum(nn.functional.pad(sq[j][None, :].expand(T + 4, hop), (0, 0, j, 3 - j)) for j in range(4))
    # ola block r is the padded signal's block r + 2; keep demucs' window after center=True trimming
    start = half + hop // 2 * 3 - 2 * hop
    y = ola.reshape(B, S, C2 // 2, -1)[..., start:start + length]
    e = env.reshape(-1)[start + 2 * hop:start + 2 * hop + length]
    return y / torch.where(e > 1e-11, e, torch.ones_like(e))


class LowMemoryAttention(nn.MultiheadAttention):
    """nn.MultiheadAttention computed one head at a time, with the same weights and result.

    The transformer in HT-Demucs attends over 2688 spectrogram positions, so all eight heads'
    attention matrices at once take about 0.5 GB; one at a time, ONNX Runtime reuses one
    head's buffers for the next. That keeps the model inside a phone browser's memory.
    """

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        assert attn_mask is None and key_padding_mask is None and self.batch_first
        E, H = self.embed_dim, self.num_heads
        d = E // H
        w, b = self.in_proj_weight, self.in_proj_bias
        q = nn.functional.linear(query, w[:E], b[:E]) * d ** -0.5
        k = nn.functional.linear(key, w[E:2 * E], b[E:2 * E])
        v = nn.functional.linear(value, w[2 * E:], b[2 * E:])
        heads = []
        for h in range(H):
            sl = slice(h * d, (h + 1) * d)
            att = torch.softmax(q[..., sl] @ k[..., sl].transpose(-1, -2), dim=-1)
            heads.append(att @ v[..., sl])
        return self.out_proj(torch.cat(heads, dim=-1)), None


def low_memory(model):
    """Swap in the memory-saving attention (same weights, same results)."""
    for mod in model.modules():
        if type(mod) is nn.MultiheadAttention:
            assert mod._qkv_same_embed_dim and mod.in_proj_bias is not None and mod.bias_k is None
            mod.__class__ = LowMemoryAttention
    return model


def load_model(name: str | None, tiny: bool):
    if tiny:
        from demucs.htdemucs import HTDemucs

        torch.manual_seed(0)
        # Same structure as htdemucs, much smaller: keeps the test fixture a few MB.
        model = HTDemucs(sources=["drums", "bass", "other", "vocals"], channels=8, depth=4, t_layers=1,
                         t_heads=2, segment=1, bottom_channels=0, dconv_comp=4)
        # Untrained weights would make the transformer near-identity; make every path matter.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        return model.eval()
    from demucs.pretrained import get_model

    bag = get_model(name)
    models = getattr(bag, "models", [bag])
    if len(models) != 1:
        raise SystemExit(f"{name} is a bag of {len(models)} models; export a single model such as htdemucs")
    return models[0].eval()


def spec_of(model, mix):
    """The model's own spectrogram of a chunk, complex as channels: [B, 2C, F, T]."""
    z = model._spec(mix)
    return model._magnitude(z)


def export(model, out: Path, name: str, fp16: bool) -> dict:
    L = int(model.segment * model.samplerate)
    mix = torch.randn(1, model.audio_channels, L) * 0.1
    mag = spec_of(model, mix)
    core = Core(low_memory(copy.deepcopy(model))).eval()
    # The fused attention kernel PyTorch uses in eval mode has no ONNX equivalent.
    torch.backends.mha.set_fastpath_enabled(False)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            core, (mix, mag), str(path), input_names=["mix", "spec"], output_names=["sources"],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    import onnx
    import onnxruntime as ort

    # Fold the shape arithmetic (int64 and float64 casts that ONNX Runtime Web lacks) into
    # constants now, so the browser can load the graph without its optimizer, which costs
    # hundreds of MB at peak on a phone.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(path)
    ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    m = onnx.load(str(path))
    # Weights go in a separate file the browser downloads once and caches. With --fp16 that file
    # holds float16 values (half the download), widened back to float32 in the browser, so the
    # graph itself still computes in float32 everywhere (WebGPU or WebAssembly).
    data_name = f"{name}.onnx.data"
    onnx.save_model(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name, size_threshold=1024)
    m = onnx.load(str(path), load_external_data=False)
    for t in m.graph.initializer:
        if any(e.key == "location" for e in t.external_data) and t.data_type != onnx.TensorProto.FLOAT:
            raise SystemExit(f"external tensor {t.name} isn't float32; the browser loader assumes it is")
    raw = np.fromfile(out / data_name, dtype="<f4")
    params = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    if raw.size > params + 2_000_000:  # folded positional embeddings are fine; the DFT tables aren't
        raise SystemExit(f"the weights file has {raw.size} values but the model {params}: a computed table was stored")
    if fp16:
        raw.astype("<f2").tofile(out / f"{name}.weights.f16")
        raw.astype("<f2").astype("<f4").tofile(out / data_name)  # what the browser will see, for check()
    else:
        raw.tofile(out / f"{name}.weights.f32")
    meta = {
        "name": name,
        "graph": f"{name}.onnx",
        "external_data_path": data_name,
        "weights": f"{name}.weights.{'f16' if fp16 else 'f32'}",
        "weights_format": "f16" if fp16 else "f32",
        "sources": list(model.sources),
        "samplerate": model.samplerate,
        "audio_channels": model.audio_channels,
        "segment_samples": L,
        "nfft": model.nfft,
        "hop": model.hop_length,
        "frames": int(mag.shape[-1]),
        "outputs": ["sources"],  # [1, S, C, L] audio (older exports returned spectrograms)
        "weights_mb": round((out / f"{name}.weights.{'f16' if fp16 else 'f32'}").stat().st_size / 1048576, 1),
    }
    digest = hashlib.sha256()
    for f in (meta["graph"], meta["weights"]):
        digest.update((out / f).read_bytes())
    meta["version"] = digest.hexdigest()[:16]
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def check(model, out: Path, name: str) -> float:
    """Run the ONNX core + PyTorch's own masking/inverse spectrogram; compare with model(mix)."""
    import onnxruntime as ort

    L = int(model.segment * model.samplerate)
    torch.manual_seed(1)
    mix = torch.randn(1, model.audio_channels, L) * 0.1
    sess = ort.InferenceSession(str(out / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    mag = spec_of(model, mix)
    (x,) = sess.run(None, {"mix": mix.numpy(), "spec": mag.numpy()})
    with torch.no_grad():
        ref = model(mix)
    err = float((torch.from_numpy(x) - ref).abs().max() / ref.abs().max())
    return err


def reference(model, out: Path, name: str) -> None:
    """A short test signal and what demucs.apply.apply_model makes of it (no shifts, 25% overlap)."""
    from demucs.apply import apply_model

    sr = model.samplerate
    n = int(sr * model.segment * 2.4)  # several overlapping chunks plus a short last one
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    left = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.1 * rng.standard_normal(n) * (np.sin(2 * np.pi * 2 * t) > 0.6)
    right = 0.3 * np.sin(2 * np.pi * 220 * t + 0.4) + 0.05 * rng.standard_normal(n)
    wav = torch.tensor(np.stack([left, right]), dtype=torch.float32)
    ref = wav.mean(0)
    norm = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        sources = apply_model(model, norm[None], shifts=0, split=True, overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    wav.numpy().astype("<f4").tofile(out / f"{name}.ref_in.f32")
    sources.numpy().astype("<f4").tofile(out / f"{name}.ref_out.f32")
    meta = json.loads((out / f"{name}.json").read_text())
    meta["reference"] = {"samples": n, "channels": 2, "sources": len(model.sources)}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--tiny", action="store_true", help="small random-weight model with the same structure (tests)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name")
    p.add_argument("--fp16", action="store_true", help="ship weights as float16 (half the download)")
    p.add_argument("--reference", action="store_true", help="also write a test signal and apply_model's output")
    args = p.parse_args()
    model = load_model(args.model, args.tiny)
    name = args.name or ("htdemucs_tiny" if args.tiny else args.model)
    meta = export(model, args.out, name, args.fp16)
    err = check(model, args.out, name)
    print(json.dumps({**meta, "max_rel_error_vs_pytorch": err}, indent=2))
    tol = 1e-2 if args.fp16 else 1e-4
    if not math.isfinite(err) or err > tol:
        raise SystemExit(f"ONNX output differs from PyTorch by {err:.2e} (> {tol})")
    if args.reference:
        reference(model, args.out, name)
    # The browser builds this file from the weights file; it isn't shipped.
    (args.out / meta["external_data_path"]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
