# Configuration Guide

## Device Configuration System

NoEmbed now uses a hierarchical configuration system that allows runtime device changes without modifying the `.env` file.

### Configuration Priority (Highest to Lowest)

1. **`config/device.json`** - Runtime configuration (writable)
2. **`.env` file** - Environment configuration (read-only in container)
3. **Default values** - Hardcoded defaults

### How It Works

When the service starts:
1. First checks for `config/device.json`
2. If not found, reads from `.env` file
3. If not found, uses default values

### Changing Device Settings

#### Via Dashboard
1. Navigate to the Configuration section
2. Select CPU or CUDA from the dropdown
3. Click "Save Device Setting"
4. Restart the container

#### Via API
```bash
# Update to CUDA
curl -X POST http://localhost:7000/api/update-device \
  -H "Content-Type: application/json" \
  -d '{"device": "cuda"}'

# Update to CPU
curl -X POST http://localhost:7000/api/update-device \
  -H "Content-Type: application/json" \
  -d '{"device": "cpu"}'
```

### Configuration File Location

Device settings are stored in:
```
./config/device.json
```

Example content:
```json
{
  "device": "cuda"
}
```

### Docker Volume Mapping

Ensure the config directory is mounted in your docker-compose.yml:
```yaml
volumes:
  - ./models:/app/models
  - ./config:/app/config  # Required for writable configuration
```

### Benefits

1. **No .env file editing** - Device can be changed at runtime
2. **Persistent across restarts** - Settings saved to disk
3. **Container-safe** - Works with read-only .env files
4. **Clear priority** - Config file always overrides .env

### Checking Current Configuration

At startup, the service logs show:
```
============================================================
NoEmbed Service - Startup Configuration
============================================================

Configuration Directory: config
  - Exists: True
  - Writable: True

Device Config File: config/device.json
  - Exists: True
  - Content: {"device": "cuda"}

.env File: .env
  - Exists: True
  - Device setting: DEVICE=cpu

Final Configuration:
  - Device: cuda  <-- Using config file, not .env
```

### Troubleshooting

**Device setting not persisting:**
- Check if `./config` directory exists and is writable
- Ensure Docker volume is mounted correctly
- Check container logs for permission errors

**CUDA errors on startup:**
- Verify CUDA is available on your system
- Check GPU is accessible to Docker
- Fall back to CPU if CUDA unavailable

**Configuration not loading:**
- Delete `config/device.json` to reset to .env defaults
- Check file permissions on config directory
- Verify JSON syntax in device.json