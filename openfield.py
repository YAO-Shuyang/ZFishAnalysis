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

def rereference_xy2polar(
    x: float,
    y: float,
    x_center: float = 0.5,
    y_center: float = 0.5
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
        The x-coordinate of the reference center (default is 0.5).
    y_center : float, optional
        The y-coordinate of the reference center (default is 0.5).

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
    alpha = np.arctan2(dy, dx)

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
    y_range: tuple
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

    Returns
    -------
    occupancy_map : np.ndarray
        A 2D array representing the occupancy map.
    """
    # Create a 2D histogram to represent occupancy
    occupancy_map, _, _ = np.histogram2d(
        x, y, bins=[x_bins, y_bins], range=[x_range, y_range]
    )

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