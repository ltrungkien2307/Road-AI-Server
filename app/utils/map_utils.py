"""
Map utilities for visualizing damages on maps
Generates GeoJSON data for map visualization
"""

from typing import List, Dict, Any
import logging
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_geojson_feature(damage: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a damage record to GeoJSON Feature format
    
    Args:
        damage: Damage record from database
        
    Returns:
        GeoJSON Feature dict
    """
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [damage['longitude'], damage['latitude']]
        },
        "properties": {
            "id": str(damage.get('id', '')),
            "type": damage.get('damage_type', 'unknown'),
            "severity": damage.get('severity', 'unknown'),
            "description": damage.get('description', ''),
            "image_url": damage.get('image_url'),
            "reported_at": str(damage.get('reported_at', '')),
            "metadata": damage.get('metadata', {})
        }
    }
    
    return feature


def create_geojson_feature_collection(damages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert list of damages to GeoJSON FeatureCollection
    
    Args:
        damages: List of damage records
        
    Returns:
        GeoJSON FeatureCollection
    """
    features = [create_geojson_feature(damage) for damage in damages]
    
    collection = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_damages": len(damages),
            "damage_types": _get_damage_summary(damages),
            "severity_distribution": _get_severity_summary(damages)
        }
    }
    
    return collection


def _get_damage_summary(damages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get count of damages by type"""
    summary = {}
    for damage in damages:
        damage_type = damage.get('damage_type', 'unknown')
        summary[damage_type] = summary.get(damage_type, 0) + 1
    return summary


def _get_severity_summary(damages: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get count of damages by severity"""
    summary = {}
    for damage in damages:
        severity = damage.get('severity', 'unknown')
        summary[severity] = summary.get(severity, 0) + 1
    return summary


def create_map_cluster_data(damages: List[Dict[str, Any]], grid_size_meters: float = 50) -> Dict[str, Any]:
    """
    Group nearby damages into clusters for map performance
    
    Args:
        damages: List of damage records
        grid_size_meters: Size of grid cells in meters
        
    Returns:
        Clustered damage data
    """
    from app.services.gps_mapper import GPSMapper
    
    gps_mapper = GPSMapper()
    
    # Convert grid size to lat/lon degrees (approximate)
    # 1 degree ≈ 111 km
    degree_per_meter = 1 / 111000
    grid_size_degrees = grid_size_meters * degree_per_meter
    
    # Create grid cells
    cells = {}
    
    for damage in damages:
        lat = damage.get('latitude')
        lon = damage.get('longitude')
        
        if not lat or not lon:
            continue
        
        # Calculate grid cell
        cell_lat = int(lat / grid_size_degrees) * grid_size_degrees
        cell_lon = int(lon / grid_size_degrees) * grid_size_degrees
        cell_key = f"{cell_lat:.4f},{cell_lon:.4f}"
        
        if cell_key not in cells:
            cells[cell_key] = {
                "center": {"lat": cell_lat, "lon": cell_lon},
                "damages": []
            }
        
        cells[cell_key]["damages"].append(damage)
    
    # Convert to feature collection
    features = []
    
    for cell_key, cell_data in cells.items():
        damages_in_cell = cell_data["damages"]
        
        # Create cluster feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [cell_data["center"]["lon"], cell_data["center"]["lat"]]
            },
            "properties": {
                "cluster": True,
                "count": len(damages_in_cell),
                "damage_types": _get_damage_summary(damages_in_cell),
                "severity_distribution": _get_severity_summary(damages_in_cell),
                "damages": damages_in_cell
            }
        }
        
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_clusters": len(features),
            "total_damages": len(damages),
            "grid_size_meters": grid_size_meters
        }
    }


def generate_heatmap_data(damages: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Generate heatmap data for map visualization
    
    Args:
        damages: List of damage records
        
    Returns:
        List of [lat, lon, intensity] for heatmap
    """
    heatmap_points = []
    
    for damage in damages:
        lat = damage.get('latitude')
        lon = damage.get('longitude')
        severity = damage.get('severity', 'low')
        
        if not lat or not lon:
            continue
        
        # Convert severity to intensity (0-1)
        intensity_map = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
        intensity = intensity_map.get(severity, 0.3)
        
        heatmap_points.append([lat, lon, intensity])
    
    return heatmap_points
