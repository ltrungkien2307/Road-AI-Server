# app/services/gps_mapper.py
"""
GPS mapping service
Maps video detections to GPS coordinates and groups nearby damages
"""

import math
from typing import List, Dict, Any, Tuple
import logging

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GPSMapper:
    """
    GPS coordinate mapper and grouper
    Handles GPS interpolation and proximity-based grouping
    """
    
    def __init__(self):
        self.earth_radius_meters = 6371000  # Earth radius in meters
    
    
    def interpolate_gps(
        self,
        timestamp: float,
        gps_log: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Interpolate GPS coordinates for a specific timestamp
        
        Args:
            timestamp: Video timestamp in seconds
            gps_log: List of GPS points with timestamps
            
        Returns:
            Dict with 'lat' and 'lon' keys
        """
        try:
            if not gps_log:
                raise ValueError("GPS log is empty")
            
            # Sort GPS log by timestamp
            sorted_gps = sorted(gps_log, key=lambda x: x['timestamp'])
            
            # Find surrounding GPS points
            before = None
            after = None
            
            for gps_point in sorted_gps:
                if gps_point['timestamp'] <= timestamp:
                    before = gps_point
                elif gps_point['timestamp'] > timestamp and after is None:
                    after = gps_point
                    break
            
            # If exact match
            if before and before['timestamp'] == timestamp:
                return {'lat': before['lat'], 'lon': before['lon']}
            
            # If timestamp is before all GPS points
            if before is None and after:
                return {'lat': after['lat'], 'lon': after['lon']}
            
            # If timestamp is after all GPS points
            if after is None and before:
                return {'lat': before['lat'], 'lon': before['lon']}
            
            # Interpolate between two points
            if before and after:
                # Linear interpolation
                time_diff = after['timestamp'] - before['timestamp']
                time_ratio = (timestamp - before['timestamp']) / time_diff
                
                lat = before['lat'] + (after['lat'] - before['lat']) * time_ratio
                lon = before['lon'] + (after['lon'] - before['lon']) * time_ratio
                
                return {'lat': lat, 'lon': lon}
            
            raise ValueError(f"Cannot interpolate GPS for timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"GPS interpolation failed: {e}")
            # Return first GPS point as fallback
            if gps_log:
                return {'lat': gps_log[0]['lat'], 'lon': gps_log[0]['lon']}
            raise
    
    
    def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula
        
        Returns:
            Distance in meters
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        distance = self.earth_radius_meters * c
        
        return distance
    
    
    def group_by_proximity(
        self,
        detections: List[Dict[str, Any]],
        threshold_meters: float = None
    ) -> List[Dict[str, Any]]:
        """
        Group detections by GPS proximity
        Detections within threshold distance are considered the same damage
        
        Args:
            detections: List of detections with gps_location
            threshold_meters: Distance threshold for grouping
            
        Returns:
            List of grouped detections (one per unique damage)
        """
        try:
            if not detections:
                return []
            
            threshold = threshold_meters or settings.GPS_PROXIMITY_THRESHOLD_METERS
            
            # Filter detections with GPS data
            valid_detections = [
                d for d in detections
                if d.get('gps_location') and 
                   d['gps_location'].get('lat') and
                   d['gps_location'].get('lon')
            ]
            
            if not valid_detections:
                logger.warning("No detections with valid GPS data")
                return []
            
            # Group by damage type first (same type damages can be grouped)
            groups_by_type = {}
            for det in valid_detections:
                damage_type = det['class_name']
                if damage_type not in groups_by_type:
                    groups_by_type[damage_type] = []
                groups_by_type[damage_type].append(det)
            
            # Group each type by proximity
            final_groups = []
            
            for damage_type, type_detections in groups_by_type.items():
                # Sort by confidence (highest first)
                sorted_dets = sorted(
                    type_detections,
                    key=lambda x: x['confidence'],
                    reverse=True
                )
                
                used_indices = set()
                
                for i, det1 in enumerate(sorted_dets):
                    if i in used_indices:
                        continue
                    
                    # Start new group with this detection
                    group = [det1]
                    used_indices.add(i)
                    
                    lat1 = det1['gps_location']['lat']
                    lon1 = det1['gps_location']['lon']
                    
                    # Find nearby detections
                    for j, det2 in enumerate(sorted_dets):
                        if j in used_indices or j <= i:
                            continue
                        
                        lat2 = det2['gps_location']['lat']
                        lon2 = det2['gps_location']['lon']
                        
                        distance = self.calculate_distance(lat1, lon1, lat2, lon2)
                        
                        if distance <= threshold:
                            group.append(det2)
                            used_indices.add(j)
                    
                    # Create representative detection for group
                    representative = self._create_group_representative(group)
                    final_groups.append(representative)
            
            logger.info(f"Grouped {len(valid_detections)} detections into {len(final_groups)} unique damages")
            
            return final_groups
            
        except Exception as e:
            logger.error(f"Grouping failed: {e}")
            return detections  # Return ungrouped as fallback
    
    
    def _create_group_representative(
        self,
        group: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a representative detection from a group
        Uses highest confidence detection, but includes group metadata
        """
        if not group:
            return None
        
        # Sort by confidence
        sorted_group = sorted(group, key=lambda x: x['confidence'], reverse=True)
        
        # Use highest confidence detection as base
        representative = sorted_group[0].copy()
        
        # Calculate average GPS location (centroid)
        avg_lat = sum(d['gps_location']['lat'] for d in group) / len(group)
        avg_lon = sum(d['gps_location']['lon'] for d in group) / len(group)
        
        representative['gps_location'] = {'lat': avg_lat, 'lon': avg_lon}
        
        # Add group metadata
        representative['detection_count'] = len(group)
        representative['confidence_range'] = [
            min(d['confidence'] for d in group),
            max(d['confidence'] for d in group)
        ]
        representative['frame_range'] = [
            min(d['frame_number'] for d in group),
            max(d['frame_number'] for d in group)
        ]
        
        # Recalculate severity based on group
        # More detections of same damage = higher severity
        if len(group) >= 5:
            representative['severity'] = 'critical'
        elif len(group) >= 3:
            if representative['severity'] == 'low':
                representative['severity'] = 'medium'
            elif representative['severity'] == 'medium':
                representative['severity'] = 'high'
        
        return representative
    
    
    def calculate_center_point(
        self,
        coordinates: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Calculate geographic center of multiple coordinates
        
        Args:
            coordinates: List of (lat, lon) tuples
            
        Returns:
            Center point as (lat, lon)
        """
        if not coordinates:
            return (0, 0)
        
        avg_lat = sum(c[0] for c in coordinates) / len(coordinates)
        avg_lon = sum(c[1] for c in coordinates) / len(coordinates)
        
        return (avg_lat, avg_lon)