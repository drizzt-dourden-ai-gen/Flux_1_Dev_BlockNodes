# comfyui/custom_nodes/flux_S21-S28_block.py
# S-block control for S21–S28 (Micro-Detail / Fine Texture)

import torch

# ---------------------------------------------------------------------------
# Patch factory for S-blocks
# ---------------------------------------------------------------------------

def _make_sblock_patch(img_strength: float):
    """
    S-block patch: modifies image-space features.
    """
    def patch(args, extra_options):
        original_block = extra_options["original_block"]

        if abs(img_strength) < 1e-6:
            return original_block(args)

        patched_args = dict(args)

        if "img" in patched_args:
            patched_args["img"] = patched_args["img"] * (1.0 + img_strength)

        return original_block(patched_args)

    return patch


# ---------------------------------------------------------------------------
# S21–S28 Node
# ---------------------------------------------------------------------------

class FluxS21S28Block:
    """
    Controls S21–S28 (Micro-Detail / Fine Texture).
    One img slider per block.
    """

    @classmethod
    def INPUT_TYPES(cls):

        slider_img = {
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

        for i in range(21, 29):
            required[f"S{i}_img"] = ("FLOAT", slider_img)

        return {"required": required}

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION     = "apply"
    CATEGORY     = "flux/s_blocks"

    def apply(self, model, clip, **kwargs):

        base_model = model.clone()

        for i in range(21, 29):
            img_strength = kwargs.get(f"S{i}_img", 0.0)
            if abs(img_strength) > 1e-6:
                base_model.set_model_patch_replace(
                    _make_sblock_patch(img_strength),
                    "dit", "single_block", i
                )

        return (base_model, clip)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "FluxS21S28Block": FluxS21S28Block,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxS21S28Block": "Flux S‑Block Control (S21–S28)",
}
