"""
DictionaryPreprocessor: composes preprocessing steps with before/after visualization.
Also supports generating individual single-step images for ablation/CER testing.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from dictextractor.preprocessing.steps import (
    to_grayscale,
    deskew as _deskew,
    denoise as _denoise,
    contrast_normalization as _contrast,
    sharpen as _sharpen,
)


class DictionaryPreprocessor:
    """Preprocesses dictionary page images with optional step-by-step visualization."""

    def __init__(self, image_path: str, output_dir: str = "preprocessing_outputs"):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.original = cv2.imread(str(image_path))
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.current = self.original.copy()
        self.metadata: Dict = {
            "original_shape": self.original.shape,
            "deskew_angle": None,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_step(
        self,
        step_name: str,
        before: np.ndarray,
        after: np.ndarray,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """Save a side-by-side before/after matplotlib figure for a step."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        def _to_display(img):
            if len(img.shape) == 3 and img.shape[2] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        b_disp = _to_display(before)
        a_disp = _to_display(after)

        cmap = "gray" if len(b_disp.shape) == 2 else None
        axes[0].imshow(b_disp, cmap=cmap)
        axes[0].set_title(f"Before: {step_name}", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        cmap = "gray" if len(a_disp.shape) == 2 else None
        axes[1].imshow(a_disp, cmap=cmap)
        axes[1].set_title(f"After: {step_name}", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()

        if save:
            out = self.output_dir / f"{step_name.lower().replace(' ', '_').replace(':', '')}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved visualization: {out}")

        if show:
            plt.show()
        else:
            plt.close()

    # ------------------------------------------------------------------
    # Individual steps (update self.current in-place)
    # ------------------------------------------------------------------

    def step1_convert_to_grayscale(self) -> np.ndarray:
        before = self.current.copy()
        self.current = to_grayscale(self.current)
        self.visualize_step("Step 1: Convert to Grayscale", before, self.current)
        return self.current

    def step2_deskew(self, skip_threshold: float = 0.5) -> np.ndarray:
        before = self.current.copy()
        self.current, angle = _deskew(self.current, skip_threshold)
        self.metadata["deskew_angle"] = angle
        if angle != 0.0:
            print(f"Applied rotation correction: {angle:.2f}°")
        else:
            print("Deskew: no significant skew detected, skipped.")
        self.visualize_step("Step 2: Deskew", before, self.current)
        return self.current

    def step3_denoise(self) -> np.ndarray:
        before = self.current.copy()
        self.current = _denoise(self.current)
        self.visualize_step("Step 3: Denoise (Bilateral)", before, self.current)
        return self.current

    def step4_contrast_normalization(self) -> np.ndarray:
        before = self.current.copy()
        self.current = _contrast(self.current)
        self.visualize_step("Step 4: Contrast Normalization (CLAHE)", before, self.current)
        return self.current

    def step5_sharpen(self, amount: float = 0.5) -> np.ndarray:
        before = self.current.copy()
        self.current = _sharpen(self.current, amount)
        self.visualize_step("Step 5: Sharpen (Unsharp Mask)", before, self.current)
        return self.current

    # ------------------------------------------------------------------
    # Pipeline runner
    # ------------------------------------------------------------------

    def run_all_steps(
        self,
        deskew: bool = True,
        deskew_threshold: float = 0.5,
        denoise: bool = True,
        contrast: bool = True,
        sharpen: bool = True,
        sharpen_amount: float = 0.5,
    ) -> np.ndarray:
        """Run all preprocessing steps in sequence with configurable toggles."""
        print(f"\n{'='*60}")
        print(f"Starting preprocessing pipeline for: {self.image_path.name}")
        print(f"{'='*60}\n")

        self.step1_convert_to_grayscale()

        if deskew:
            self.step2_deskew(skip_threshold=deskew_threshold)
        else:
            print("Skipping Step 2: Deskew")

        if denoise:
            self.step3_denoise()
        else:
            print("Skipping Step 3: Denoise")

        if contrast:
            self.step4_contrast_normalization()
        else:
            print("Skipping Step 4: Contrast Normalization")

        if sharpen:
            self.step5_sharpen(amount=sharpen_amount)
        else:
            print("Skipping Step 5: Sharpen")

        self.visualize_step("Final Result: Original vs Preprocessed", self.original, self.current)

        print(f"\n{'='*60}")
        print("Preprocessing complete!")
        if self.metadata.get("deskew_angle"):
            print(f"Deskew angle applied: {self.metadata['deskew_angle']:.2f}°")
        print(f"{'='*60}\n")

        return self.current

    # ------------------------------------------------------------------
    # Save result
    # ------------------------------------------------------------------

    def save_result(self, output_path: Optional[str] = None) -> str:
        """Save the final preprocessed image to disk."""
        if output_path is None:
            out = self.output_dir / f"preprocessed_{self.image_path.name}"
        else:
            out = Path(output_path)
        cv2.imwrite(str(out), self.current)
        print(f"Saved preprocessed image to: {out}")
        return str(out)

    # ------------------------------------------------------------------
    # Ablation / CER testing
    # ------------------------------------------------------------------

    def generate_individual_preprocessed_images(
        self,
        deskew_threshold: float = 0.5,
        denoise_params: Optional[dict] = None,
        contrast_params: Optional[dict] = None,
        sharpen_amount: float = 0.5,
    ) -> Dict[str, str]:
        """
        Generate one image per preprocessing step (grayscale + ONE step each).
        Used for ablation / CER comparison experiments.

        Returns:
            Mapping of step name → output file path.
        """
        if denoise_params is None:
            denoise_params = {"d": 5, "sigma_color": 75, "sigma_space": 75}
        if contrast_params is None:
            contrast_params = {"clip_limit": 2.0, "tile_size": 8}

        output_paths: Dict[str, str] = {}

        print(f"\n{'='*60}")
        print("Generating individual preprocessed images for CER testing...")
        print(f"{'='*60}\n")

        def _gray(img):
            return to_grayscale(img) if len(img.shape) == 3 else img.copy()

        # 1. Baseline: grayscale only
        img = _gray(self.original)
        p = self.output_dir / "preprocessed_baseline_grayscale.png"
        cv2.imwrite(str(p), img)
        output_paths["baseline"] = str(p)
        print(f"1. baseline → {p.name}")

        # 2. Grayscale + deskew
        img, _ = _deskew(_gray(self.original), deskew_threshold)
        p = self.output_dir / "preprocessed_deskew_only.png"
        cv2.imwrite(str(p), img)
        output_paths["deskew"] = str(p)
        print(f"2. deskew_only → {p.name}")

        # 3. Grayscale + denoise
        img = _denoise(_gray(self.original), **denoise_params)
        p = self.output_dir / "preprocessed_denoise_only.png"
        cv2.imwrite(str(p), img)
        output_paths["denoise"] = str(p)
        print(f"3. denoise_only → {p.name}")

        # 4. Grayscale + contrast
        img = _contrast(
            _gray(self.original),
            clip_limit=contrast_params["clip_limit"],
            tile_size=contrast_params["tile_size"],
        )
        p = self.output_dir / "preprocessed_contrast_only.png"
        cv2.imwrite(str(p), img)
        output_paths["contrast"] = str(p)
        print(f"4. contrast_only → {p.name}")

        # 5. Grayscale + sharpen
        img = _sharpen(_gray(self.original), amount=sharpen_amount)
        p = self.output_dir / "preprocessed_sharpen_only.png"
        cv2.imwrite(str(p), img)
        output_paths["sharpen"] = str(p)
        print(f"5. sharpen_only → {p.name}")

        print(f"\n{'='*60}")
        print(f"Generated {len(output_paths)} images in: {self.output_dir}")
        print(f"{'='*60}\n")

        return output_paths
