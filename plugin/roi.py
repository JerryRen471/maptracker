import numpy as np


def resolve_roi(roi_size, roi_range=None):
    """Return a validated XY range and size for a BEV region."""
    size = np.asarray(roi_size, dtype=np.float64)
    if size.shape != (2,) or np.any(size <= 0):
        raise ValueError(f"roi_size must contain two positive values: {roi_size}")

    if roi_range is None:
        origin = -size / 2
        bounds = np.concatenate([origin, origin + size])
    else:
        bounds = np.asarray(roi_range, dtype=np.float64)
        if bounds.shape != (4,):
            raise ValueError(
                "roi_range must be (x_min, y_min, x_max, y_max)")
        range_size = bounds[2:] - bounds[:2]
        if np.any(range_size <= 0):
            raise ValueError(f"roi_range is invalid: {roi_range}")
        if not np.allclose(range_size, size):
            raise ValueError(
                f"roi_range size {range_size.tolist()} does not match "
                f"roi_size {size.tolist()}")

    return bounds, size


def normalize_points(points, roi_range, roi_size):
    points = np.asarray(points)
    return (points - np.asarray(roi_range)[:2]) / np.asarray(roi_size)


def denormalize_points(points, roi_range, roi_size):
    points = np.asarray(points)
    return points * np.asarray(roi_size) + np.asarray(roi_range)[:2]


def metric_to_grid(points, roi_range, roi_size):
    """Convert metric XY coordinates to grid_sample coordinates."""
    normalized = normalize_points(points, roi_range, roi_size)
    grid = normalized * 2 - 1
    grid[..., 1] *= -1
    return grid


def roi_cache_tag(roi_range):
    def encode(value):
        value = float(value)
        prefix = "m" if value < 0 else ""
        magnitude = abs(value)
        text = str(int(magnitude)) if magnitude.is_integer() else str(magnitude)
        return f"{prefix}{text.replace('.', 'p')}"

    xmin, ymin, xmax, ymax = roi_range
    return (
        f"x{encode(xmin)}_y{encode(ymin)}_"
        f"x{encode(xmax)}_y{encode(ymax)}")
