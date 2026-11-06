"""
This code provides basic methods of processing open-field data.

Heading direction (theta) is not the same as the angle determined by 
np.arctan2(y, x) (alpha) because the heading direction is defined as 
1. North = 0
2. East = 90
3. South = 180
4. West = 270

but alpha is defined by default as
1. East = 0
2. North = 90
3. West = 180
4. South = 270

so the heading direction can be calculated from alpha as follows:
    theta = (90 - alpha) % 360
or reversely, alpha can be calculated from theta as follows:
    alpha = (90 - theta) % 360
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

def rereference_xy2polar(
    x: float,
    y: float,
    x_center: float = 50,
    y_center: float = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert x, y coordinates to polar coordinates with respect to a given center.

    Parameters
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    x_center : float, optional
        The x-coordinate of the reference center (default is 50).
    y_center : float, optional
        The y-coordinate of the reference center (default is 50).

    Returns
    -------
    r : float
        The radial distance from the center to the point.
    alpha : float
        The angle in radians from the positive x-axis to the point.
    """
    # Calculate the difference from the center
    dx = x - x_center
    dy = y - y_center

    # Calculate polar coordinates
    r = np.sqrt(dx**2 + dy**2)
    alpha = np.arctan2(dy, dx) % (2 * np.pi)  # Ensure alpha is in [0, 2π)

    return r, alpha

def rereference_polar2xy(
    r: float,
    alpha: float,
    x_center: float = 0.5,
    y_center: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert polar coordinates to x, y coordinates with respect to a given center.

    Parameters
    ----------
    r : float
        The radial distance from the center to the point.
    alpha : float
        The angle in radians from the positive x-axis to the point.
    x_center : float, optional
        The x-coordinate of the reference center (default is 0.5).
    y_center : float, optional
        The y-coordinate of the reference center (default is 0.5).

    Returns
    -------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    """
    # Calculate Cartesian coordinates
    x = r * np.cos(alpha) + x_center
    y = r * np.sin(alpha) + y_center

    return x, y

def get_occupancy2D(
    x: np.ndarray,
    y: np.ndarray,
    x_bins: int,
    y_bins: int,
    x_range: tuple,
    y_range: tuple,
    is_clipped: bool = True
) -> np.ndarray:
    """
    Calculate the 2D occupancy map based on x and y coordinates. It counts the 
    number of occurrences of points in each bin defined by the specified ranges 
    and number of bins.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the points.
    y : np.ndarray
        The y-coordinates of the points.
    x_bins : int
        The number of bins along the x-axis.
    y_bins : int
        The number of bins along the y-axis.
    x_range : tuple
        The range of x values (min, max).
    y_range : tuple
        The range of y values (min, max).
    is_clipped : bool
        If True, clip the x and y values to the specified ranges before calculating

    Returns
    -------
    occupancy_map : np.ndarray
        A 2D array representing the occupancy map.
    """
    assert y_range[0] < y_range[1], "y_range[0] must be less than y_range[1]"
    assert x_range[0] < x_range[1], "x_range[0] must be less than x_range[1]"
    
    if is_clipped:
        x = np.clip(x, x_range[0], x_range[1]-1e-8)  # Clip to avoid edge case at upper bound
        y = np.clip(y, y_range[0], y_range[1]-1e-8)  # Clip to avoid edge case at upper bound
    
    # Create a 2D histogram to represent occupancy
    occupancy_map, _, _ = np.histogram2d(
        x, y, bins=[x_bins, y_bins], range=[x_range, y_range]
    )

    return occupancy_map

def get_occupancy_radial(
    x: np.ndarray,
    y: np.ndarray,
    r_bins: int,
    r_range: tuple[float, float] = (0, 15+1e-3),
    is_area_normalized: bool = False,
    is_clipped: bool = True,
    **ref_kwargs
) -> np.ndarray:
    """
    Calculate the radial occupancy map based on x and y coordinates. It counts the
    number of occurrences of points in each radial bin defined by the specified range
    and number of bins.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the points.
    y : np.ndarray
        The y-coordinates of the points.
    r_bins : int
        The number of radial bins.
    r_range : tuple
        The range of radial distances (min, max).
    is_area_normalized : bool
        If True, normalize the occupancy map by the area of each radial bin.
    is_clipped : bool
        If True, clip the radial distances to the specified range before calculating
        the occupancy map.
    **ref_kwargs : dict
        Additional keyword arguments for the rereference_xy2polar function, such as
        `x_center` and `y_center`.

    Returns
    -------
    occupancy_map : np.ndarray
        A 1D array representing the radial occupancy map.
    """
    # reference x, y to polar coordinates
    r, _ = rereference_xy2polar(x, y, **ref_kwargs)

    # Validate r_range
    if r_range[0] < 0:
        raise ValueError("r_range[0] must be non-negative.")
    if r_range[1] <= r_range[0]:
        raise ValueError("r_range[1] must be greater than r_range[0].")
    
    # Create a 1D histogram to represent radial occupancy
    if is_clipped:
        r = np.clip(r, r_range[0], r_range[1]-1e-8)  # Clip to avoid edge case at upper bound
    occupancy_map, _ = np.histogram(r, bins=r_bins, range=r_range)
    
    # Normalize by area if requested
    if is_area_normalized:
        # Calculate the area of each radial bin
        bin_edges = np.linspace(r_range[0], r_range[1], r_bins + 1)
        bin_areas = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        
        # Avoid division by zero
        assert np.all(bin_areas > 0), "Bin areas must be positive."
        
        # Normalize occupancy map by area
        occupancy_map = occupancy_map / bin_areas

    return occupancy_map

def get_occupancy_angular(
    x: np.ndarray,
    y: np.ndarray,
    angle_bins: int,
    angle_range: tuple[float, float] = (0, 2 * np.pi),
    radial_threshold: Optional[float] = 7.5,
    alternatives: str = "greater",
    **ref_kwargs
) -> np.ndarray:
    """
    Calculate the angular occupancy map based on x and y coordinates. It counts the
    number of occurrences of points in each angular bin defined by the specified
    number of bins.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the points.
    y : np.ndarray
        The y-coordinates of the points.
    angle_bins : int
        The number of angular bins.
    angle_range : tuple
        The range of angles (min, max) in radians.
    radial_threshold : float, optional
        If provided, only points with a radial distance greater than this threshold
        will be considered for the angular occupancy calculation.
    alternatives : str
        Specifies the comparison for the radial threshold. Can be "greater" or "less".
        It only applies if `radial_threshold` is provided.
    **ref_kwargs : dict
        Additional keyword arguments for the rereference_xy2polar function, such as
        `x_center` and `y_center`.
    
    Returns
    -------
    occupancy_map : np.ndarray
        A 1D array representing the angular occupancy map.
    """
    # reference x, y to polar coordinates
    r, alpha = rereference_xy2polar(x, y, **ref_kwargs)

    # Filter points based on radial threshold if provided
    if radial_threshold is not None:
        assert alternatives in ["greater", "less"], \
            "Invalid value for 'alternatives'. Must be 'greater' or 'less'."
        if alternatives == "greater":
            mask = r > radial_threshold
        elif alternatives == "less":
            mask = r < radial_threshold
        else:
            raise ValueError(
                "Invalid value for 'alternatives'. Must be 'greater' or 'less'."
            )
        alpha = alpha[mask]
    else:
        # If no radial threshold is provided, use all points
        alpha = alpha

    # Create a 1D histogram to represent angular occupancy
    occupancy_map, _ = np.histogram(alpha, bins=angle_bins, range=angle_range)

    return occupancy_map

def reorient(
    angle: np.ndarray,
) -> np.ndarray:
    """
    Reorient the angle, either from alpha to theta (heading direction) or
    from theta to alpha (standard angle from the positive x-axis).

    Parameters
    ----------
    angle : np.ndarray
        The angle in radians from the positive x-axis (alpha) or the heading 
        direction (theta).

    Returns
    -------
    np.ndarray
        The reoriented angle in radians, either from heading direction to 
        standard angle or vice versa.
    """
    # Convert angle to degrees for easier manipulation
    angle_deg = np.degrees(angle)
    
    # Convert
    theta_deg = (90 - angle_deg) % 360
    
    # Convert back to radians
    theta_rad = np.radians(theta_deg)

    return theta_rad


@dataclass
class OpenField:
    shape: str
    xrange: tuple[float, float]
    yrange: tuple[float, float]
    landmarks: List[str]
    landmark_ranges: List[tuple[float, float, float, float]]
    coord_sys: str = 'radial'
    
    def __post_init__(self):
        # Validate shape
        assert self.shape in ['circle', 'square'], \
            "Shape must be either 'circle' or 'square'."
        
        # Validate coordinate system
        assert self.coord_sys in ['radial', 'cartesian'], \
            "Coordinate system must be either 'radial' or 'cartesian'."
        
        # Validate landmark ranges
        for lr in self.landmark_ranges:
            assert len(lr) == 4, \
                "Each landmark range must be a tuple of (xmin, xmax, ymin, ymax) " \
                "or (ang_l, ang_r, radial_l, radial_r)."
            assert lr[0] < lr[1], \
                "In each landmark range, xmin/xang_l must be less than xmax/ang_r."
            assert lr[2] < lr[3], \
                "In each landmark range, ymin/radial_l must be less than ymax/radial_r."
        
        assert len(self.landmarks) == len(self.landmark_ranges), \
            "The number of landmarks must match the number of landmark ranges."
            
        self.lm = dict(zip(self.landmarks, self.landmark_ranges))
        
    def get_landmark(self, x: float, y: float) -> Optional[str]:
        """
        Determine which landmark the point (x, y) belongs to.

        Parameters
        ----------
        x : float
            The x-coordinate of the point.
        y : float
            The y-coordinate of the point.

        Returns
        -------
        str or None
            The name of the landmark if the point is within a landmark range,
            otherwise None.
        """
        if self.coord_sys == 'cartesian':
            for landmark, (xmin, xmax, ymin, ymax) in self.lm.items():
                if xmin <= x < xmax and ymin <= y < ymax:
                    return landmark
            return None
        elif self.coord_sys == 'radial':
            for landmark, (ang_l, ang_r, radial_l, radial_r) in self.lm.items():
                r, alpha = rereference_xy2polar(x, y)
                if radial_l <= r < radial_r and ang_l <= alpha < ang_r:
                    return landmark
            return None
        else:
            raise ValueError("Invalid coordinate system. Must be 'radial' or 'cartesian'.")


OF_v1 = OpenField(
    shape='circle',
    xrange=(35, 65),
    yrange=(35, 65),
    landmarks=['gravel', 'star', 'polygon', 'square', 'wave', 'leave', 'triangle', 'teeth', 'center'],
    landmark_ranges=[
        (np.radians(0), np.radians(45), 7.5, 15),   
        (np.radians(45), np.radians(90), 7.5, 15),
        (np.radians(90), np.radians(135), 7.5, 15),
        (np.radians(135), np.radians(180), 7.5, 15),
        (np.radians(180), np.radians(225), 7.5, 15),
        (np.radians(225), np.radians(270), 7.5, 15),
        (np.radians(270), np.radians(315), 7.5, 15),
        (np.radians(315), np.radians(360), 7.5, 15),
        (np.radians(0), np.radians(360), 0, 7.5),
    ],
    coord_sys='radial'
)