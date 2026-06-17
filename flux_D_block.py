# comfyui/custom_nodes/flux_D_block.py
# D-block control for D0–D18 (Semantic Control)

import torch

# ---------------------------------------------------------------------------
# Patch factory for D-blocks
# ---------------------------------------------------------------------------

def _make_dblock_patch(strength_txt: float, strength_img: float):
    """
    Double-block patch: semantic control via txt and img stream scaling.
    """
    def patch(args, extra_options):
        original_block = extra_options["original_block"]

        if abs(strength_txt) < 1e-6 and abs(strength_img) < 1e-6:
            return original_block(args)

        patched_args = dict(args)

        if abs(strength_txt) > 1e-6 and "txt" in patched_args:
            patched_args["txt"] = patched_args["txt"] * (1.0 + strength_txt)

        if abs(strength_img) > 1e-6 and "img" in patched_args:
            patched_args["img"] = patched_args["img"] * (1.0 + strength_img)

        return original_block(patched_args)

    return patch


# ---------------------------------------------------------------------------
# D-Block Control Node
# ---------------------------------------------------------------------------

class FluxDBlockControl:
    """
    Controls all 19 D-blocks (D0–D18).
    Two sliders per block: txt stream (semantic) and img stream (spatial/structural).
    """

    @classmethod
    def INPUT_TYPES(cls):

        slider = {
            "default": 0.0,
            "min":    -0.02,
            "max":     0.02,
            "step":    0.001,
            "display": "slider",
        }

        required = {
            "model": ("MODEL",),
            "clip":  ("CLIP",),
        }

        for i in range(19):
            required[f"D{i}_txt"] = ("FLOAT", slider)
            required[f"D{i}_img"] = ("FLOAT", slider)

        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION     = "apply"
    CATEGORY     = "flux/d_blocks"

    def apply(self, model, clip, **kwargs):

        base_model = model.clone()

        for i in range(19):
            strength_txt = kwargs.get(f"D{i}_txt", 0.0)
            strength_img = kwargs.get(f"D{i}_img", 0.0)
            if abs(strength_txt) > 1e-6 or abs(strength_img) > 1e-6:
                base_model.set_model_patch_replace(
                    _make_dblock_patch(strength_txt, strength_img),
                    "dit", "double_block", i
                )

        return (base_model, clip)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FluxDBlockControl": FluxDBlockControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxDBlockControl": "Flux D‑Block Control (D0–D18)",
}
