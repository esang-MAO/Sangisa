"""Export HT-Demucs to ONNX for the in-browser engine (site/engine/demucs.js).

Only the network goes into the ONNX graph. The spectrogram in front of it and the inverse
spectrogram behind it (complex numbers, which ONNX Runtime Web handles poorly) stay in
JavaScript, as does chunking the song. So the graph is:

    inputs   mix   float32 [1, 2, L]           one chunk of audio (L = the model's training length)
             spec  float32 [1, 4, 2048, T]     its spectrogram, complex as channels (demucs' "CaC")
    outputs  spec_out float32 [1, S, 4, 2048, T]   per-source spectrograms (to invert in JS)
             time_out float32 [1, S, 2, L]         per-source time-branch output (added after inverting)

Usage:
    python scripts/export_demucs_onnx.py --model htdemucs --out site/models          # pretrained
    python scripts/export_demucs_onnx.py --tiny --out tests/engine/fixtures --reference  # test fixture

Writes <out>/<name>.onnx (the graph), <name>.weights.f16 or .f32 (the weights, which the graph
reads as the external data file <name>.onnx.data) and <name>.json (shapes, sources, file names).
With --reference also a short test signal and the PyTorch apply_model output, for parity tests.
"""

from __future__ import annotations

import argparse
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
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
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
        return x, xt


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
    core = Core(model).eval()
    # The fused attention kernel PyTorch uses in eval mode has no ONNX equivalent.
    torch.backends.mha.set_fastpath_enabled(False)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            core, (mix, mag), str(path), input_names=["mix", "spec"], output_names=["spec_out", "time_out"],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    import onnx

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
        "weights_mb": round((out / f"{name}.weights.{'f16' if fp16 else 'f32'}").stat().st_size / 1048576, 1),
    }
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
    so, to = sess.run(None, {"mix": mix.numpy(), "spec": mag.numpy()})
    with torch.no_grad():
        ref = model(mix)
        zout = model._mask(None, torch.from_numpy(so))
        x = model._ispec(zout, L) + torch.from_numpy(to)
    err = float((x - ref).abs().max() / ref.abs().max())
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
