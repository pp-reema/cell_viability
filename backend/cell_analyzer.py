import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from skimage import measure, morphology, filters, segmentation
from skimage.feature import peak_local_max
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

class CellViabilityAnalyzer:
    """
    Advanced cell viability analyzer optimized for blue-stained microscope images.
    Handles cells with visible boundaries and varying stain intensity.
    """

    def __init__(self,
                 dead_cell_threshold=0.50,
                 min_cell_area=300,
                 max_cell_area=15000,
                 circularity_threshold=0.25):
        """
        Initialize analyzer.

        Args:
            dead_cell_threshold: Cells darker than this are considered dead (0-1)
            min_cell_area: Minimum cell size in pixels
            max_cell_area: Maximum cell size in pixels
            circularity_threshold: Minimum circularity (0-1, 1=perfect circle)
        """
        self.dead_threshold = dead_cell_threshold
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        self.circularity_threshold = circularity_threshold

    def preprocess_image(self, image):
        """Enhance image quality and reduce noise."""
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)

        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return enhanced

    def detect_cell_regions(self, image):
        """
        Detect all cell regions using adaptive thresholding and morphology.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding to handle uneven illumination
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=5
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Remove small noise
        binary = morphology.remove_small_objects(binary.astype(bool),
                                                  min_size=self.min_cell_area // 2)

        return binary.astype(np.uint8) * 255

    def segment_individual_cells(self, image, binary_mask):
        """
        Segment individual cells using watershed algorithm.
        """
        # Distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        # Smooth distance transform
        dist_smooth = filters.gaussian(dist_transform, sigma=2)

        # Find peaks (cell centers)
        local_max = peak_local_max(
            dist_smooth,
            min_distance=15,
            threshold_rel=0.2,
            labels=binary_mask
        )

        # Create markers
        markers = np.zeros_like(binary_mask, dtype=np.int32)
        for idx, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = idx

        # Dilate markers slightly
        markers = ndimage.grey_dilation(markers, size=(3, 3))

        # Watershed segmentation
        markers = segmentation.watershed(-dist_transform, markers, mask=binary_mask)

        return markers

    def extract_cell_features(self, image, markers):
        """
        Extract features for each segmented cell.
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Get region properties
        regions = measure.regionprops(markers, intensity_image=image)

        cells = []

        for region in regions:
            # Filter by size
            if region.area < self.min_cell_area or region.area > self.max_cell_area:
                continue

            # Calculate circularity (area / convex_area)
            if region.convex_area > 0:
                circularity = region.area / region.convex_area
            else:
                circularity = 0

            # Skip non-circular objects (likely debris or cell clumps)
            if circularity < self.circularity_threshold:
                continue

            # Extract cell region
            minr, minc, maxr, maxc = region.bbox
            cell_mask = markers == region.label

            # Calculate color features
            cell_rgb = image[cell_mask]
            cell_hsv = hsv[cell_mask]

            # HSV features (H=hue, S=saturation, V=value/brightness)
            mean_hue = np.mean(cell_hsv[:, 0])
            mean_saturation = np.mean(cell_hsv[:, 1]) / 255.0
            mean_value = np.mean(cell_hsv[:, 2]) / 255.0  # Brightness

            # RGB features
            mean_rgb = np.mean(cell_rgb, axis=0)
            mean_blue = mean_rgb[2] / 255.0

            # Calculate blue ratio (blue vs other channels)
            blue_ratio = mean_blue / (np.mean(mean_rgb[:2]) / 255.0 + 1e-6)

            cells.append({
                'region': region,
                'label': region.label,
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,
                'circularity': circularity,
                'mean_value': mean_value,
                'mean_saturation': mean_saturation,
                'mean_blue': mean_blue,
                'blue_ratio': blue_ratio,
                'mean_rgb': mean_rgb,
                'hsv_pixels': cell_hsv
            })

        return cells

    def classify_cells(self, cells, method='adaptive'):
        """
        Classify cells as live or dead.
        """
        if len(cells) == 0:
            return [], []

        live_cells = []
        dead_cells = []

        for cell in cells:
            hsv_pixels = cell['hsv_pixels']
            if len(hsv_pixels) > 0:
                blue_hue_mask = (hsv_pixels[:, 0] >= 90) & (hsv_pixels[:, 0] <= 140)
                saturated_mask = hsv_pixels[:, 1] > 30
                stained_pixels_count = np.sum(blue_hue_mask & saturated_mask)
                is_stained = stained_pixels_count / len(hsv_pixels) > 0.4
            else:
                is_stained = False

            if not is_stained:
                live_cells.append(cell)
            elif method == 'adaptive':
                is_dead = self._classify_single_cell_adaptive(cell)
                if is_dead:
                    dead_cells.append(cell)
                else:
                    live_cells.append(cell)
            else:
                is_dead = self._classify_single_cell_threshold(cell)
                if is_dead:
                    dead_cells.append(cell)
                else:
                    live_cells.append(cell)

        return live_cells, dead_cells

    def _classify_single_cell_threshold(self, cell):
        """Classify a single cell using fixed threshold."""
        is_dead = (
            cell['mean_value'] < self.dead_threshold and
            cell['mean_saturation'] > 0.2
        )
        return is_dead

    def _classify_single_cell_adaptive(self, cell):
        """Classify a single cell using K-Means on its brightness."""
        brightness = np.array([pixel[2] for pixel in cell['hsv_pixels']]).reshape(-1, 1)

        if len(brightness) < 5:
            return False

        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(brightness)
            centers = kmeans.cluster_centers_.flatten()

            dead_cluster = 0 if centers[0] < centers[1] else 1
            dead_pixels_count = np.sum(labels == dead_cluster)

            return dead_pixels_count > len(labels) * 0.7

        except Exception as e:
            print(f"KMeans clustering failed for a cell: {e}")
            return False

    def calculate_statistics(self, live_cells, dead_cells):
        """Calculate comprehensive statistics."""
        total = len(live_cells) + len(dead_cells)

        if total == 0:
            return {
                'total_cells': 0,
                'live_cells': 0,
                'dead_cells': 0,
                'live_percentage': 0.0,
                'dead_percentage': 0.0,
                'viability': 0.0
            }

        live_count = len(live_cells)
        dead_count = len(dead_cells)
        live_pct = (live_count / total) * 100
        dead_pct = (dead_count / total) * 100

        return {
            'total_cells': total,
            'live_cells': live_count,
            'dead_cells': dead_count,
            'live_percentage': live_pct,
            'dead_percentage': dead_pct,
            'viability': live_pct
        }

    def _create_detection_overlay(self, image, live_cells, dead_cells):
        """Create overlay with all cell contours."""
        overlay = image.copy()

        all_cells = live_cells + dead_cells
        for cell in all_cells:
            minr, minc, maxr, maxc = cell['bbox']
            color = (0, 255, 0) if cell in live_cells else (255, 0, 0)

            cv2.rectangle(overlay, (minc, minr), (maxc, maxr), color, 2)

            cy, cx = [int(c) for c in cell['centroid']]
            cv2.circle(overlay, (cx, cy), 5, color, -1)

        return overlay

    def _create_classification_overlay(self, image, live_cells, dead_cells):
        """Create overlay with color-coded filled masks."""
        overlay = image.copy().astype(float)

        for cell in dead_cells:
            mask = np.zeros(image.shape[:2], dtype=bool)
            minr, minc, maxr, maxc = cell['bbox']
            mask[minr:maxr, minc:maxc] = cell['region'].image
            overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5

        for cell in live_cells:
            mask = np.zeros(image.shape[:2], dtype=bool)
            minr, minc, maxr, maxc = cell['bbox']
            mask[minr:maxr, minc:maxc] = cell['region'].image
            overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

        return overlay.astype(np.uint8)

    def analyze(self, image_path, method='adaptive', visualize=False):
        """
        Complete analysis pipeline.
        """
        # Load image
        if isinstance(image_path, str):
            image = np.array(Image.open(image_path).convert('RGB'))
            print(f"Loaded image: {image.shape}")
        else:
            image = image_path
            print(f"Processing image: {image.shape}")

        # Preprocess
        print("Preprocessing image...")
        enhanced = self.preprocess_image(image)

        # Detect cells
        print("Detecting cell regions...")
        binary_mask = self.detect_cell_regions(enhanced)

        # Segment individual cells
        print("Segmenting individual cells...")
        markers = self.segment_individual_cells(enhanced, binary_mask)

        # Extract features
        print("Extracting cell features...")
        cells = self.extract_cell_features(enhanced, markers)
        print(f"   Found {len(cells)} valid cells")

        # Classify
        print(f"Classifying cells using '{method}' method...")
        live_cells, dead_cells = self.classify_cells(cells, method=method)

        # Calculate statistics
        print("Calculating statistics...")
        stats = self.calculate_statistics(live_cells, dead_cells)

        # Create overlays
        overlay = self._create_detection_overlay(image, live_cells, dead_cells)
        classification = self._create_classification_overlay(image, live_cells, dead_cells)

        # Print summary
        print("\n" + "="*70)
        print("CELL VIABILITY ANALYSIS RESULTS")
        print("="*70)
        print(f"{'Total Cells Detected:':<30} {stats['total_cells']:>5d}")
        print(f"{'Live Cells:':<30} {stats['live_cells']:>5d} ({stats['live_percentage']:>5.1f}%)")
        print(f"{'Dead Cells:':<30} {stats['dead_cells']:>5d} ({stats['dead_percentage']:>5.1f}%)")
        print("-"*70)
        print(f"{'CELL VIABILITY:':<30} {stats['viability']:>5.2f}%")
        print("="*70 + "\n")

        return {
            'statistics': stats,
            'overlay': overlay,
            'classification': classification,
            'live_cells': live_cells,
            'dead_cells': dead_cells,
            'all_cells': cells
        }
