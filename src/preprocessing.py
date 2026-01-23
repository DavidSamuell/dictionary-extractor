"""
PaddleOCR preprocessing and visualization pipeline for dictionary extraction.
Implements all preprocessing steps from PaddleOCRPlan.md with before/after visualization.
"""

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional
import math


class DictionaryPreprocessor:
    """Preprocesses dictionary page images with visualization."""

    def __init__(self, image_path: str, output_dir: str = "preprocessing_outputs"):
        """
        Initialize preprocessor with input image.

        Args:
            image_path: Path to dictionary page image
            output_dir: Directory to save visualization outputs
        """
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load original image
        self.original = cv2.imread(str(image_path))
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Keep track of current processed image
        self.current = self.original.copy()

        # Store metadata
        self.metadata = {
            "original_shape": self.original.shape,
            "deskew_angle": None,
        }

    def visualize_step(
        self,
        step_name: str,
        before: np.ndarray,
        after: np.ndarray,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Visualize before/after for a preprocessing step.

        Args:
            step_name: Name of the preprocessing step
            before: Image before processing
            after: Image after processing
            save: Whether to save the visualization
            show: Whether to display the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Convert BGR to RGB for matplotlib
        before_rgb = (
            cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
            if len(before.shape) == 3 and before.shape[2] == 3
            else before
        )
        after_rgb = (
            cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
            if len(after.shape) == 3 and after.shape[2] == 3
            else after
        )

        # Handle grayscale images
        if len(before_rgb.shape) == 2:
            axes[0].imshow(before_rgb, cmap="gray")
        else:
            axes[0].imshow(before_rgb)
        axes[0].set_title(f"Before: {step_name}", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        if len(after_rgb.shape) == 2:
            axes[1].imshow(after_rgb, cmap="gray")
        else:
            axes[1].imshow(after_rgb)
        axes[1].set_title(f"After: {step_name}", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{step_name.lower().replace(' ', '_')}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization: {output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def step1_convert_to_grayscale(self) -> np.ndarray:
        """
        Step 1: Convert to grayscale.
        Better for OCR - reduces noise and data size.
        """
        before = self.current.copy()

        # Convert to grayscale if not already
        if len(self.current.shape) == 3:
            # BGR or RGB to grayscale
            self.current = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        # else: already grayscale

        after = self.current.copy()
        self.visualize_step("Step 1: Convert to Grayscale", before, after)

        return self.current

    def step2_deskew(self, skip_threshold: float = 0.5) -> np.ndarray:
        """
        Step 2: Deskew the image using Hough line detection.

        Args:
            skip_threshold: Skip correction if angle < this value (degrees)
        """
        before = self.current.copy()

        # Work with grayscale (already converted in step 1)
        gray = (
            self.current
            if len(self.current.shape) == 2
            else cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        )

        # Binarize for better line detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect lines using Hough transform
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            # Calculate angles
            angles = []
            for rho, theta in lines[:, 0]:
                # Convert to degrees, normalize to [-45, 45]
                angle = (theta * 180 / np.pi) - 90
                # Filter out vertical/horizontal outliers
                if -45 < angle < 45:
                    angles.append(angle)

            if angles:
                # Use median angle for robustness
                skew_angle = np.median(angles)
                self.metadata["deskew_angle"] = skew_angle

                print(f"Detected skew angle: {skew_angle:.2f}°")

                # Only rotate if angle is significant
                if abs(skew_angle) > skip_threshold:
                    # Get image center and rotation matrix
                    h, w = self.current.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

                    # Rotate with white background (255 for grayscale)
                    border_value = (
                        255 if len(self.current.shape) == 2 else (255, 255, 255)
                    )
                    self.current = cv2.warpAffine(
                        self.current,
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=border_value,
                    )
                    print(f"Applied rotation correction: {skew_angle:.2f}°")
                else:
                    print(
                        f"Skew angle too small ({skew_angle:.2f}°), skipping correction"
                    )
            else:
                print("No valid angles detected, skipping deskew")
                self.metadata["deskew_angle"] = 0.0
        else:
            print("No lines detected, skipping deskew")
            self.metadata["deskew_angle"] = 0.0

        after = self.current.copy()
        self.visualize_step("Step 2: Deskew", before, after)

        return self.current

    def step3_denoise(self, kernel_size: int = 3) -> np.ndarray:
        """
        Step 3: Light denoising using bilateral filter.
        Preserves edges while reducing noise.

        Args:
            kernel_size: Size of the filter kernel (odd number)
        """
        before = self.current.copy()

        # Bilateral filter: reduces noise while keeping edges sharp
        # d: diameter of pixel neighborhood
        # sigmaColor: filter sigma in color space
        # sigmaSpace: filter sigma in coordinate space
        self.current = cv2.bilateralFilter(
            self.current, d=5, sigmaColor=75, sigmaSpace=75
        )

        after = self.current.copy()
        self.visualize_step("Step 3: Denoise (Bilateral)", before, after)

        return self.current

    def step4_contrast_normalization(
        self, clip_limit: float = 2.0, tile_size: int = 8
    ) -> np.ndarray:
        """
        Step 4: Contrast normalization using CLAHE.
        Helps with small Cyrillic characters.

        Args:
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization
        """
        before = self.current.copy()

        # Apply CLAHE directly to grayscale image
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
        )
        self.current = clahe.apply(self.current)

        after = self.current.copy()
        self.visualize_step("Step 4: Contrast Normalization (CLAHE)", before, after)

        return self.current

    def step5_sharpen(self, amount: float = 1.0) -> np.ndarray:
        """
        Step 5: Optional mild sharpening.
        Use carefully to avoid halos.

        Args:
            amount: Sharpening strength (0.5-2.0 recommended)
        """
        before = self.current.copy()

        # Create unsharp mask
        gaussian = cv2.GaussianBlur(self.current, (0, 0), 2.0)
        self.current = cv2.addWeighted(self.current, 1.0 + amount, gaussian, -amount, 0)

        after = self.current.copy()
        self.visualize_step("Step 5: Sharpen (Unsharp Mask)", before, after)

        return self.current

    def run_all_steps(
        self,
        deskew: bool = True,
        deskew_threshold: float = 0.5,
        denoise: bool = True,
        contrast: bool = True,
        sharpen: bool = True,
        sharpen_amount: float = 0.5,
    ) -> np.ndarray:
        """
        Run all preprocessing steps in sequence.

        Args:
            deskew: Whether to apply deskew
            deskew_threshold: Minimum angle (degrees) to apply deskew
            denoise: Whether to apply denoising
            contrast: Whether to apply contrast normalization
            sharpen: Whether to apply sharpening
            sharpen_amount: Sharpening strength

        Returns:
            Final preprocessed image
        """
        print(f"\n{'='*60}")
        print(f"Starting preprocessing pipeline for: {self.image_path.name}")
        print(f"{'='*60}\n")

        # Step 1: Convert to grayscale
        print("Running Step 1: Convert to Grayscale...")
        self.step1_convert_to_grayscale()

        # Step 2: Deskew (optional)
        if deskew:
            print("\nRunning Step 2: Deskew...")
            self.step2_deskew(skip_threshold=deskew_threshold)
        else:
            print("\nSkipping Step 2: Deskew")

        # Step 3: Denoise (optional)
        if denoise:
            print("\nRunning Step 3: Denoise...")
            self.step3_denoise()
        else:
            print("\nSkipping Step 3: Denoise")

        # Step 4: Contrast normalization (optional)
        if contrast:
            print("\nRunning Step 4: Contrast Normalization...")
            self.step4_contrast_normalization()
        else:
            print("\nSkipping Step 4: Contrast Normalization")

        # Step 5: Sharpen (optional)
        if sharpen:
            print("\nRunning Step 5: Sharpen...")
            self.step5_sharpen(amount=sharpen_amount)
        else:
            print("\nSkipping Step 5: Sharpen")

        # Final comparison
        self.visualize_step(
            "Final Result: Original vs Preprocessed", self.original, self.current
        )

        print(f"\n{'='*60}")
        print("Preprocessing complete!")
        print(f"Visualizations saved to: {self.output_dir}")
        if self.metadata["deskew_angle"] is not None:
            print(f"Deskew angle applied: {self.metadata['deskew_angle']:.2f}°")
        print(f"{'='*60}\n")

        return self.current

    def save_result(self, output_path: Optional[str] = None) -> str:
        """Save the final preprocessed image."""
        if output_path is None:
            output_path = self.output_dir / f"preprocessed_{self.image_path.name}"
        else:
            output_path = Path(output_path)

        cv2.imwrite(str(output_path), self.current)
        print(f"Saved preprocessed image to: {output_path}")
        return str(output_path)

    def generate_individual_preprocessed_images(
        self,
        deskew_threshold: float = 0.5,
        denoise_params: dict = None,
        contrast_params: dict = None,
        sharpen_amount: float = 0.5,
    ) -> Dict[str, str]:
        """
        Generate individual preprocessed images for each step to test CER impact.
        Each image only has grayscale + ONE additional preprocessing step applied.

        Args:
            deskew_threshold: Minimum angle for deskew correction
            denoise_params: Parameters for denoising (d, sigmaColor, sigmaSpace)
            contrast_params: Parameters for CLAHE (clip_limit, tile_size)
            sharpen_amount: Sharpening strength

        Returns:
            Dictionary mapping step name to output file path
        """
        if denoise_params is None:
            denoise_params = {"d": 5, "sigmaColor": 75, "sigmaSpace": 75}
        if contrast_params is None:
            contrast_params = {"clip_limit": 2.0, "tile_size": 8}

        output_paths = {}

        print(f"\n{'='*60}")
        print("Generating individual preprocessed images for CER testing...")
        print(f"{'='*60}\n")

        # 1. Baseline: Just grayscale
        print("1. Generating baseline (grayscale only)...")
        img = self.original.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output_path = self.output_dir / f"preprocessed_baseline_grayscale.png"
        cv2.imwrite(str(output_path), img)
        output_paths["baseline"] = str(output_path)
        print(f"   Saved: {output_path}")

        # 2. Grayscale + Deskew
        print("\n2. Generating grayscale + deskew...")
        img = self.original.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply deskew
        gray = img.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 < angle < 45:
                    angles.append(angle)

            if angles:
                skew_angle = np.median(angles)
                if abs(skew_angle) > deskew_threshold:
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    img = cv2.warpAffine(
                        img,
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=255,
                    )
                    print(f"   Applied deskew: {skew_angle:.2f}°")

        output_path = self.output_dir / f"preprocessed_deskew_only.png"
        cv2.imwrite(str(output_path), img)
        output_paths["deskew"] = str(output_path)
        print(f"   Saved: {output_path}")

        # 3. Grayscale + Denoise
        print("\n3. Generating grayscale + denoise...")
        img = self.original.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bilateralFilter(
            img,
            d=denoise_params["d"],
            sigmaColor=denoise_params["sigmaColor"],
            sigmaSpace=denoise_params["sigmaSpace"],
        )
        output_path = self.output_dir / f"preprocessed_denoise_only.png"
        cv2.imwrite(str(output_path), img)
        output_paths["denoise"] = str(output_path)
        print(f"   Saved: {output_path}")

        # 4. Grayscale + Contrast (CLAHE)
        print("\n4. Generating grayscale + contrast normalization...")
        img = self.original.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=contrast_params["clip_limit"],
            tileGridSize=(contrast_params["tile_size"], contrast_params["tile_size"]),
        )
        img = clahe.apply(img)
        output_path = self.output_dir / f"preprocessed_contrast_only.png"
        cv2.imwrite(str(output_path), img)
        output_paths["contrast"] = str(output_path)
        print(f"   Saved: {output_path}")

        # 5. Grayscale + Sharpen
        print("\n5. Generating grayscale + sharpen...")
        img = self.original.copy()
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.0 + sharpen_amount, gaussian, -sharpen_amount, 0)
        output_path = self.output_dir / f"preprocessed_sharpen_only.png"
        cv2.imwrite(str(output_path), img)
        output_paths["sharpen"] = str(output_path)
        print(f"   Saved: {output_path}")

        print(f"\n{'='*60}")
        print("Individual preprocessed images generated!")
        print(f"Total images: {len(output_paths)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        return output_paths


def main():
    """Demo/test the preprocessing pipeline."""
    # Use the test dictionary page
    test_image = Path(__file__).parent.parent / "assets" / "test-dict-page.png"

    if not test_image.exists():
        print(f"Error: Test image not found at {test_image}")
        print("Please provide a path to a dictionary page image.")
        return

    # Initialize preprocessor
    preprocessor = DictionaryPreprocessor(
        str(test_image), output_dir="preprocessing_outputs"
    )

    # Generate individual preprocessed images for CER testing
    print("=" * 70)
    print("PART 1: Generating individual preprocessed images for CER testing")
    print("=" * 70)
    individual_paths = preprocessor.generate_individual_preprocessed_images(
        deskew_threshold=0.5, sharpen_amount=0.5
    )

    print("\n" + "=" * 70)
    print("PART 2: Running full preprocessing pipeline with all steps combined")
    print("=" * 70)

    # Run all preprocessing steps combined
    preprocessed = preprocessor.run_all_steps(
        deskew=True,
        deskew_threshold=0.5,  # Only rotate if > 0.5 degrees
        denoise=True,
        contrast=True,
        sharpen=True,
        sharpen_amount=0.5,  # Mild sharpening
    )

    # Save the combined result
    combined_path = preprocessor.save_result()

    print("\n" + "=" * 70)
    print("SUMMARY: All preprocessing outputs generated!")
    print("=" * 70)
    print("\nIndividual preprocessing outputs (for CER testing):")
    for step, path in individual_paths.items():
        print(f"  - {step}: {Path(path).name}")

    print(f"\nCombined preprocessing output:")
    print(f"  - all_steps: {Path(combined_path).name}")

    print(f"\nAll outputs saved to: preprocessing_outputs/")
    print("\nNext steps:")
    print("1. Run extraction on each individual preprocessed image")
    print("2. Run evaluation to compare CER for each preprocessing method")
    print("3. Determine which preprocessing step(s) provide the best results")


if __name__ == "__main__":
    main()
