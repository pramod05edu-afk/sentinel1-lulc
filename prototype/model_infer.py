"""Load the trained checkpoint and run LULC inference on dB S1 tiles.

The checkpoint (saved by the training notebook) carries the architecture/encoder
config and the exact channel standardization (ch_mean/ch_std), so inference
reproduces the training preprocessing: 3 channels [VV, VH, VV-VH], standardized.
"""
import numpy as np
import torch
import segmentation_models_pytorch as smp

from classes import N_CLASSES


class LULCModel:
    def __init__(self, ckpt_path, device=None, per_scene_norm=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ck.get("cfg", {})
        arch = cfg.get("arch", "unet")
        encoder = cfg.get("encoder", "resnet34")
        kw = dict(encoder_name=encoder, encoder_weights=None,
                  in_channels=3, classes=cfg.get("n_classes", N_CLASSES))
        self.model = (smp.Unet(**kw) if arch == "unet" else smp.DeepLabV3Plus(**kw))
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()
        self.ch_mean = np.asarray(ck["ch_mean"], np.float32).reshape(3, 1, 1)
        self.ch_std = np.asarray(ck["ch_std"], np.float32).reshape(3, 1, 1)
        self.tile = cfg.get("img_size", 256)
        # Per-scene standardization aligns live GEE composites to the training
        # input distribution and fixes the SEN12MS -> live domain shift that
        # otherwise collapses predictions to one class. Set False to use the
        # checkpoint's stored training statistics instead.
        self.per_scene_norm = per_scene_norm

    def _scene_stats(self, s1_db):
        """Per-channel mean/std for [VV,VH,VV-VH], from the whole scene or the ckpt."""
        if not self.per_scene_norm:
            return self.ch_mean, self.ch_std
        vv, vh = s1_db[0], s1_db[1]
        x = np.nan_to_num(np.stack([vv, vh, vv - vh], 0).astype(np.float32), nan=0.0)
        mean = np.nanmean(x, axis=(1, 2)).reshape(3, 1, 1).astype(np.float32)
        std = (np.nanstd(x, axis=(1, 2)) + 1e-6).reshape(3, 1, 1).astype(np.float32)
        return mean, std

    @staticmethod
    def _to_input(s1_db, mean, std):
        """(2,H,W) dB -> standardized (3,H,W) [VV,VH,VV-VH]."""
        vv, vh = s1_db[0], s1_db[1]
        x = np.nan_to_num(np.stack([vv, vh, vv - vh], 0).astype(np.float32), nan=0.0)
        return (x - mean) / std

    @torch.no_grad()
    def predict(self, s1_db, overlap=True):
        """(2,S,S) dB -> (S,S) uint8 label map.

        Normalization statistics are computed once on the full scene, so every
        tile shares them. With ``overlap=True`` (default) tiles overlap by half
        and their softmax probabilities are blended with a centre-weighted
        window: a pixel near one tile's border is decided mostly by the tile
        that sees it in the middle. Non-overlapping tiling (``overlap=False``,
        the original behaviour) leaves visible seams along tile edges, because
        the network sees little context near a border.
        """
        if overlap:
            return self._predict_overlap(s1_db)
        _, H, W = s1_db.shape
        t = self.tile
        mean, std = self._scene_stats(s1_db)
        out = np.zeros((H, W), np.uint8)
        for r in range(0, H, t):
            for c in range(0, W, t):
                tile = s1_db[:, r:r + t, c:c + t]
                th, tw = tile.shape[1], tile.shape[2]
                if th < t or tw < t:                       # pad edge tile
                    tile = np.pad(tile, ((0, 0), (0, t - th), (0, t - tw)), mode="edge")
                x = torch.from_numpy(self._to_input(tile, mean, std))[None].to(self.device)
                pred = self.model(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
                out[r:r + t, c:c + t] = pred[:th, :tw]
        return out

    @torch.no_grad()
    def _predict_overlap(self, s1_db):
        """Sliding-window inference, stride = tile/2, centre-weighted blending."""
        _, H, W = s1_db.shape
        t = self.tile
        s = t // 2
        mean, std = self._scene_stats(s1_db)
        # Pad by half a tile on every side (plus up to the next stride multiple),
        # so every real pixel lies in the central part of at least one window.
        ph, pw = (-H) % s, (-W) % s
        pad = ((0, 0), (s, s + ph), (s, s + pw))
        mode = "reflect" if min(H, W) > s + max(ph, pw) else "edge"
        x_full = np.pad(s1_db, pad, mode=mode)
        PH, PW = x_full.shape[1], x_full.shape[2]

        win = np.hanning(t + 2)[1:-1].astype(np.float32)     # strictly > 0
        weight = torch.from_numpy(np.outer(win, win)).to(self.device)
        acc = wsum = None
        for r in range(0, PH - t + 1, s):
            for c in range(0, PW - t + 1, s):
                tile = x_full[:, r:r + t, c:c + t]
                x = torch.from_numpy(self._to_input(tile, mean, std))[None].to(self.device)
                prob = torch.softmax(self.model(x), 1)[0] * weight          # (C,t,t)
                if acc is None:
                    acc = torch.zeros((prob.shape[0], PH, PW), device=self.device)
                    wsum = torch.zeros((PH, PW), device=self.device)
                acc[:, r:r + t, c:c + t] += prob
                wsum[r:r + t, c:c + t] += weight
        labels = (acc / wsum.clamp_min(1e-8)).argmax(0)
        return labels[s:s + H, s:s + W].cpu().numpy().astype(np.uint8)

    def predict_series(self, stack):
        """(12,2,S,S) -> (12,S,S) uint8 label maps."""
        return np.stack([self.predict(stack[i]) for i in range(stack.shape[0])])
