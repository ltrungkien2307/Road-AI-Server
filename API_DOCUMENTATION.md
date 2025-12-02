# Road-AI-Server: Video Processing with GPS-Mapped Damages

## 📋 Overview

Road-AI-Server xử lý video kèm GPS log để detect road damages và gắn chúng lên bản đồ với hình ảnh crop từ video.

## 🔄 Processing Workflow

```
1. Mobile App: Quay video + collect GPS log
2. Upload video lên Cloudinary
3. Gửi POST /api/ai/process-video (video_url + gps_log)
   ↓
Road-AI-Server:
4. Download video từ Cloudinary
5. Extract frames (1fps) + map GPS coordinates
6. Run YOLO detection trên từng frame
7. Crop detection area từ frame
8. Upload cropped image lên Cloudinary
9. Group detections by GPS proximity
10. Create damage records (GPS + cropped image)
11. Update task status
   ↓
Frontend/Mobile:
12. GET /api/ai/results/{task_id}/map → GeoJSON
13. Display damages trên map với image markers
```

## 📡 API Endpoints

### Submit Video for Processing
```
POST /api/ai/process-video

Request:
{
  "task_id": "task_123",
  "video_url": "https://res.cloudinary.com/.../video.mp4",
  "company_id": "company_123",
  "gps_log": [
    {
      "timestamp": 0.0,
      "lat": 10.8231,
      "lon": 106.6297,
      "speed": 5.2,
      "accuracy": 5.0
    },
    // ... more GPS points
  ]
}

Response:
{
  "job_id": "job_abc123",
  "status": "queued",
  "message": "Video processing task queued successfully",
  "estimated_time_minutes": 5
}
```

### Check Processing Status
```
GET /api/ai/status/{job_id}

Response:
{
  "job_id": "job_abc123",
  "status": "processing",
  "progress": 65,
  "message": "Processing frame 650/1000",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z"
}
```

### Get Results (Simple Format)
```
GET /api/ai/results/{task_id}

Response:
{
  "task_id": "task_123",
  "damage_count": 5,
  "damages": [
    {
      "id": "damage_1",
      "type": "pothole",
      "severity": "high",
      "latitude": 10.8231,
      "longitude": 106.6297,
      "image_url": "https://res.cloudinary.com/.../damage_crop.jpg",
      "description": "Pothole detected with 85% confidence",
      "metadata": {
        "confidence": 0.85,
        "frame_number": 120,
        "bbox": [100, 200, 300, 400]
      }
    }
  ]
}
```

### Get Results (GeoJSON for Map)
```
GET /api/ai/results/{task_id}/map

Response:
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [106.6297, 10.8231]
      },
      "properties": {
        "id": "damage_1",
        "type": "pothole",
        "severity": "high",
        "description": "...",
        "image_url": "https://res.cloudinary.com/.../damage_crop.jpg",
        "metadata": {...}
      }
    }
  ],
  "metadata": {
    "total_damages": 5,
    "damage_types": {
      "pothole": 3,
      "crack": 2
    },
    "severity_distribution": {
      "high": 2,
      "medium": 2,
      "low": 1
    }
  }
}
```

### Get Heatmap Data
```
GET /api/ai/results/{task_id}/heatmap

Response:
{
  "task_id": "task_123",
  "heatmap_data": [
    [10.8231, 106.6297, 1.0],    // [lat, lon, intensity]
    [10.8232, 106.6298, 0.7],
    [10.8233, 106.6299, 0.5]
  ],
  "damage_count": 3,
  "metadata": {
    "format": "[lat, lon, intensity]",
    "intensity_scale": {
      "critical": 1.0,
      "high": 0.7,
      "medium": 0.5,
      "low": 0.3
    }
  }
}
```

## 🖼️ Image Cropping Logic

### Quá Trình Crop:
1. **Bounding Box Detection**: YOLO trả về [x1, y1, x2, y2] trong frame
2. **Expand Context**: Mở rộng bbox 10% để có context xung quanh
3. **Crop from Frame**: Cắt ảnh từ frame gốc
4. **Upload to Cloudinary**: Upload cropped image với folder `damages/{job_id}`
5. **Store URL**: Lưu URL vào damage record

### Benefits:
- ✅ Hình ảnh crop nhỏ, tải nhanh hơn
- ✅ Focus trên vùng damage
- ✅ Tiết kiệm storage (không lưu toàn frame)
- ✅ Có thể zoom vào damage detail trên map

## 🗺️ Map Integration

### React Map Component Example:
```typescript
// Fetch damages in GeoJSON format
const response = await fetch(`/api/ai/results/${taskId}/map`);
const geojson = await response.json();

// Add to map (Leaflet, Mapbox, etc.)
geojson.features.forEach(feature => {
  const { properties, geometry } = feature;
  
  // Create marker with damage info
  const marker = L.marker([geometry.coordinates[1], geometry.coordinates[0]])
    .bindPopup(`
      <div>
        <h3>${properties.type}</h3>
        <p>Severity: ${properties.severity}</p>
        <img src="${properties.image_url}" width="200">
        <p>${properties.description}</p>
      </div>
    `)
    .addTo(map);
});
```

### Heatmap Example:
```typescript
// Fetch heatmap data
const response = await fetch(`/api/ai/results/${taskId}/heatmap`);
const data = await response.json();

// Add heatmap layer (using Leaflet.heat)
const heat = L.heatLayer(data.heatmap_data, {
  radius: 25,
  blur: 15,
  max: 1.0,
  minOpacity: 0.3
}).addTo(map);
```

## 🎯 Configuration

### Video Processing Settings (app/config.py):
```python
FRAME_EXTRACTION_FPS = 1              # Extract 1 frame per second
GPS_PROXIMITY_THRESHOLD_METERS = 10.0  # Group damages within 10m
MIN_CONFIDENCE_FOR_DAMAGE = 0.5        # Min confidence threshold
MODEL_CONFIDENCE_THRESHOLD = 0.4       # YOLO confidence threshold
```

### Severity Calculation:
```python
SEVERITY_THRESHOLDS = {
    "critical": {"area_percent": 15, "confidence": 0.8},
    "high": {"area_percent": 10, "confidence": 0.7},
    "medium": {"area_percent": 5, "confidence": 0.6},
    "low": {"area_percent": 0, "confidence": 0.4}
}
```

Severity được tính dựa trên:
1. **Diện tích**: % chiếm trong frame
2. **Confidence**: Độ chính xác từ model
3. **Detection Count**: Số lần phát hiện (khi grouping)

## 📊 Damage Grouping

Khi có nhiều detection gần nhau:
- **Proximiti-based grouping**: Detections trong 10m coi là 1 damage
- **Centroid GPS**: Dùng trung bình GPS của group
- **Highest confidence**: Lấy detection có confidence cao nhất làm representative
- **Severity elevation**: Nếu group có 3+ detections, severity tăng

Ví dụ:
```
Detection 1: pothole, confidence 0.6, severity medium
Detection 2: pothole, confidence 0.7, severity medium
Detection 3: pothole, confidence 0.65, severity medium

→ Grouped: 1 pothole, avg GPS, confidence [0.6-0.7], severity HIGH (3 detections)
```

## 🚀 Performance Optimization

### Batch Processing:
- YOLO runs batch inference (tất cả frames cùng lúc)
- Nhanh hơn 10x so với single frame processing
- Memory-efficient streaming

### Frame Extraction:
- FPS configurable (default 1fps)
- GPS interpolation cho frames giữa GPS points
- Smart grid extraction (tránh blur/dark frames)

### Image Handling:
- Crop only detection area (not full frame)
- Auto cleanup temporary files
- Cloudinary folder organization: `damages/{job_id}/{image_uuid}.jpg`

## 🔧 Development Setup

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables (.env):
```env
SUPABASE_URL=your_url
SUPABASE_SERVICE_ROLE_KEY=your_key
CLOUDINARY_CLOUD_NAME=your_name
CLOUDINARY_API_KEY=your_key
CLOUDINARY_API_SECRET=your_secret
CELERY_BROKER_URL=redis://localhost:6379/0
```

### Run Server:
```bash
# Development
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# With Celery workers
celery -A app.tasks.celery_tasks worker -l info --concurrency=1
```

## 📱 Mobile Integration

### Submit Video with GPS:
```typescript
// mobile/services/tasks.ts
await submitTaskVideo(
  taskId,
  videoUrl,           // Cloudinary URL
  gpsLog              // GPS log array
);

// Backend automatically:
// 1. Forwards to Road-AI-Server
// 2. Waits for processing
// 3. Creates damage records
// 4. Updates task status
```

### Display Results:
```typescript
// Fetch map data
const damages = await fetch(`/api/ai/results/${taskId}/map`);
const geojson = await damages.json();

// Render map with damage markers
renderMapWithDamages(geojson);
```

## ✅ Testing Checklist

- [ ] Video upload to Cloudinary
- [ ] GPS log format validation
- [ ] Frame extraction with GPS interpolation
- [ ] YOLO detection accuracy
- [ ] Image cropping & Cloudinary upload
- [ ] Damage grouping logic
- [ ] GeoJSON generation
- [ ] Map display with markers
- [ ] Heatmap rendering
- [ ] Damage image preview on click
- [ ] Task status update
- [ ] Performance with large videos (30+ min)
