import base64
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


BBox = List[int]


class CurrencyDetector:
    """
    Offline-first counterfeit screening pipeline for Indian Rs 500 notes.

    The checks are heuristic and designed for a hackathon-grade prototype:
    they combine note-layout priors with classic OpenCV measurements and
    optional local EasyOCR model inference. They should be calibrated with
    real counterfeit and genuine note image sets before operational use.
    """

    SERIAL_PATTERN = re.compile(r"^[0-9A-Z]{1,3}\s?[0-9A-Z]{1,3}\s?[0-9]{5,7}$")

    def __init__(self) -> None:
        self.weights = {
            "microprint": 0.25,
            "security_thread": 0.30,
            "serial_number": 0.25,
            "uv_features": 0.20,
        }
        self._ocr_reader = None
        self._ocr_error: Optional[str] = None

    def analyze_note(self, image: np.ndarray) -> Dict[str, Any]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return self._error_response("Input image is empty or invalid.")

        try:
            bgr = self._ensure_bgr(image)
            resized, scale = self._resize_for_analysis(bgr)
            note_box = self._locate_note(resized)
            x1, y1, x2, y2 = note_box
            note = resized[y1:y2, x1:x2]

            checks = {
                "microprint": self._analyze_microprint(note, note_box),
                "security_thread": self._verify_security_thread(note, note_box),
                "serial_number": self._validate_serial_number(note, note_box),
                "uv_features": self._simulate_uv_features(note, note_box),
            }

            for check in checks.values():
                check["bbox"] = self._scale_bbox(check["bbox"], scale)

            score = self._score(checks)
            verdict = "likely_authentic" if score >= 70 else "manual_review_required"
            if score < 45:
                verdict = "high_risk_counterfeit"

            return {
                "success": True,
                "denomination": "INR_500",
                "authenticity_score": score,
                "verdict": verdict,
                "note_bbox": self._scale_bbox(list(note_box), scale),
                "checks": checks,
                "warnings": self._warnings(checks),
            }
        except Exception as exc:
            return self._error_response(f"Analysis failed: {exc}")

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError("Unsupported image format.")

    def _resize_for_analysis(self, image: np.ndarray, max_width: int = 1200) -> Tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        if width <= max_width:
            return image.copy(), 1.0
        scale = max_width / float(width)
        resized = cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _locate_note(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 130)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = image.shape[:2]
        min_area = width * height * 0.12
        candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(h, 1)
            if area >= min_area and 1.6 <= aspect <= 3.3:
                candidates.append((area, (x, y, x + w, y + h)))

        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

        pad_x = int(width * 0.04)
        pad_y = int(height * 0.08)
        return pad_x, pad_y, width - pad_x, height - pad_y

    def _analyze_microprint(self, note: np.ndarray, note_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        h, w = note.shape[:2]
        region = self._relative_crop(note, 0.08, 0.18, 0.58, 0.43)
        bbox = self._relative_bbox(note_box, 0.08, 0.18, 0.58, 0.43)

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(clahe, 70, 170)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        laplacian_variance = float(cv2.Laplacian(clahe, cv2.CV_64F).var())

        texture_score = self._normalize(edge_density, 0.035, 0.145) * 0.55
        texture_score += self._normalize(laplacian_variance, 55.0, 360.0) * 0.45
        passed = texture_score >= 0.48

        return {
            "label": "Microprint Analysis",
            "status": "passed" if passed else "failed",
            "bbox": bbox,
            "confidence": round(texture_score, 3),
            "metrics": {
                "edge_density": round(edge_density, 4),
                "laplacian_variance": round(laplacian_variance, 2),
                "region_size": [int(w * 0.50), int(h * 0.25)],
            },
        }

    def _verify_security_thread(self, note: np.ndarray, note_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        h, w = note.shape[:2]
        search = self._relative_crop(note, 0.46, 0.08, 0.66, 0.92)
        bbox = self._relative_bbox(note_box, 0.46, 0.08, 0.66, 0.92)

        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        green_lower = np.array([35, 35, 35], dtype=np.uint8)
        green_upper = np.array([95, 255, 255], dtype=np.uint8)
        blue_lower = np.array([90, 35, 35], dtype=np.uint8)
        blue_upper = np.array([135, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, green_lower, green_upper),
            cv2.inRange(hsv, blue_lower, blue_upper),
        )
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        search_h, search_w = search.shape[:2]
        best_height = 0
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        colored_pixels = int(np.count_nonzero(mask))

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            slender = ch / max(cw, 1)
            if ch > best_height and slender >= 2.0:
                best_height = ch
                best_bbox = (x, y, x + cw, y + ch)

        continuity = best_height / float(search_h)
        color_ratio = colored_pixels / float(search_h * search_w)
        score = self._normalize(continuity, 0.34, 0.78) * 0.75
        score += self._normalize(color_ratio, 0.0025, 0.045) * 0.25
        passed = score >= 0.52

        if best_bbox:
            ox1, oy1, _, _ = bbox
            x1, y1, x2, y2 = best_bbox
            final_bbox = [ox1 + x1, oy1 + y1, ox1 + x2, oy1 + y2]
        else:
            final_bbox = bbox

        return {
            "label": "Security Thread Verification",
            "status": "passed" if passed else "failed",
            "bbox": final_bbox,
            "confidence": round(score, 3),
            "metrics": {
                "thread_continuity": round(continuity, 3),
                "green_blue_pixel_ratio": round(color_ratio, 4),
                "search_region": [int(w * 0.20), int(h * 0.84)],
            },
        }

    def _validate_serial_number(self, note: np.ndarray, note_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        region = self._relative_crop(note, 0.55, 0.70, 0.97, 0.95)
        bbox = self._relative_bbox(note_box, 0.55, 0.70, 0.97, 0.95)
        serial_text = ""
        ocr_confidence = 0.0
        ocr_error = None

        try:
            reader = self._get_ocr_reader()
            if reader is None:
                ocr_error = self._ocr_error or "EasyOCR reader is unavailable."
            else:
                enhanced = self._prepare_serial_for_ocr(region)
                results = reader.readtext(enhanced, detail=1, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                if results:
                    ordered = sorted(results, key=lambda item: item[0][0][0])
                    serial_text = "".join(str(item[1]).upper().replace(" ", "") for item in ordered)
                    ocr_confidence = float(np.mean([float(item[2]) for item in ordered]))
        except Exception as exc:
            ocr_error = str(exc)

        pattern_ok = bool(self.SERIAL_PATTERN.match(serial_text))
        scaling_score = self._serial_scaling_score(region)
        score = (1.0 if pattern_ok else 0.0) * 0.68 + scaling_score * 0.32
        passed = score >= 0.55

        metrics: Dict[str, Any] = {
            "recognized_text": serial_text,
            "ocr_confidence": round(ocr_confidence, 3),
            "pattern_valid": pattern_ok,
            "progressive_font_scaling_score": round(scaling_score, 3),
        }
        if ocr_error:
            metrics["ocr_note"] = ocr_error

        return {
            "label": "Serial Number Pattern Validation",
            "status": "passed" if passed else "failed",
            "bbox": bbox,
            "confidence": round(score, 3),
            "metrics": metrics,
        }

    def _simulate_uv_features(self, note: np.ndarray, note_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
        region = self._relative_crop(note, 0.08, 0.08, 0.40, 0.88)
        bbox = self._relative_bbox(note_box, 0.08, 0.08, 0.40, 0.88)

        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced = cv2.merge([enhanced_l, a_channel, b_channel])
        simulated_uv = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        gray_before = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(simulated_uv, cv2.COLOR_BGR2GRAY)
        contrast_gain = float(gray_after.std() - gray_before.std())
        highlight_mask = cv2.adaptiveThreshold(
            gray_after,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            -4,
        )
        highlight_ratio = float(np.count_nonzero(highlight_mask)) / float(highlight_mask.size)
        score = self._normalize(contrast_gain, 3.0, 20.0) * 0.48
        score += self._normalize(highlight_ratio, 0.08, 0.33) * 0.52
        passed = score >= 0.42

        return {
            "label": "UV Feature Simulation",
            "status": "passed" if passed else "failed",
            "bbox": bbox,
            "confidence": round(score, 3),
            "metrics": {
                "clahe_contrast_gain": round(contrast_gain, 2),
                "fluorescent_highlight_ratio": round(highlight_ratio, 3),
                "simulation": "CLAHE-enhanced visible-light approximation of UV response",
            },
            "preview_png_base64": self._preview_png(simulated_uv),
        }

    def _get_ocr_reader(self) -> Any:
        if self._ocr_reader is not None:
            return self._ocr_reader
        if self._ocr_error is not None:
            return None
        try:
            import easyocr
        except Exception as exc:
            self._ocr_error = f"easyocr package could not be imported: {exc}"
            return None
        try:
            download_enabled = os.getenv("EASYOCR_DOWNLOAD_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
            self._ocr_reader = easyocr.Reader(["en"], gpu=False, download_enabled=download_enabled, verbose=False)
            return self._ocr_reader
        except Exception as exc:
            self._ocr_error = (
                "EasyOCR local model files were not found. Install/cache the English model once, "
                f"then run offline. Original error: {exc}"
            )
            return None

    def _prepare_serial_for_ocr(self, region: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def _serial_scaling_score(self, region: np.ndarray) -> float:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights: List[int] = []
        region_h, region_w = region.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= region_w * 0.20 and region_h * 0.10 <= h <= region_h * 0.85:
                heights.append(h)

        if len(heights) < 4:
            return 0.25

        heights = sorted(heights)
        spread = (max(heights) - min(heights)) / max(float(max(heights)), 1.0)
        spread_score = self._normalize(spread, 0.05, 0.35)
        consistency = 1.0 - min(float(np.std(heights)) / max(float(np.mean(heights)), 1.0), 1.0)
        return max(0.0, min(1.0, spread_score * 0.55 + consistency * 0.45))

    def _score(self, checks: Dict[str, Dict[str, Any]]) -> int:
        weighted = 0.0
        for key, weight in self.weights.items():
            check = checks[key]
            base = 1.0 if check["status"] == "passed" else 0.0
            confidence = float(check.get("confidence", 0.0))
            weighted += weight * (base * 0.70 + confidence * 0.30)
        return int(round(weighted * 100))

    def _warnings(self, checks: Dict[str, Dict[str, Any]]) -> List[str]:
        warnings = []
        for key, check in checks.items():
            if check["status"] == "failed":
                warnings.append(f"{check['label']} failed and should be manually inspected.")
            metrics = check.get("metrics", {})
            if "ocr_note" in metrics:
                warnings.append(metrics["ocr_note"])
        return warnings

    def _relative_crop(self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        h, w = image.shape[:2]
        ax1, ay1, ax2, ay2 = self._relative_coords(w, h, x1, y1, x2, y2)
        return image[ay1:ay2, ax1:ax2]

    def _relative_bbox(
        self,
        note_box: Tuple[int, int, int, int],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> BBox:
        nx1, ny1, nx2, ny2 = note_box
        w = nx2 - nx1
        h = ny2 - ny1
        ax1, ay1, ax2, ay2 = self._relative_coords(w, h, x1, y1, x2, y2)
        return [nx1 + ax1, ny1 + ay1, nx1 + ax2, ny1 + ay2]

    def _relative_coords(self, w: int, h: int, x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int, int, int]:
        ax1 = max(0, min(w - 1, int(w * x1)))
        ay1 = max(0, min(h - 1, int(h * y1)))
        ax2 = max(ax1 + 1, min(w, int(w * x2)))
        ay2 = max(ay1 + 1, min(h, int(h * y2)))
        return ax1, ay1, ax2, ay2

    def _scale_bbox(self, bbox: BBox, scale: float) -> BBox:
        if scale == 1.0:
            return [int(v) for v in bbox]
        return [int(round(v / scale)) for v in bbox]

    def _normalize(self, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    def _preview_png(self, image: np.ndarray) -> str:
        try:
            preview = cv2.resize(image, (240, max(1, int(240 * image.shape[0] / image.shape[1]))), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".png", preview)
            if not ok:
                return ""
            return base64.b64encode(buffer).decode("ascii")
        except Exception:
            return ""

    def _error_response(self, message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "authenticity_score": 0,
            "verdict": "analysis_error",
            "error": message,
            "checks": {},
            "warnings": [message],
        }
