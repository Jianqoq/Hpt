# Conv Benchmark

::: chartjs Conv2d f32 Performance (width = 256, height = 256, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": ["64", "96", "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", "576", "608", "640", "672", "704", "736", "768"],
    "datasets": [
        {
        "label": "Torch (mkldnn + direct)",
        "data": [18.706, 25.780, 32.960, 40.384, 48.180, 55.959, 65.984, 70.528, 77.862, 85.401, 92.009, 99.589, 105.72, 109.05, 114.86, 122.00, 129.14, 134.25, 141.98, 145.11, 154.19, 161.20, 166.91],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Hpt (direct)",
        "data": [10.972, 16.498, 22.311, 28.565, 33.554, 38.697, 44.794, 50.640, 57.125, 64.182, 71.065, 77.754, 84.005, 90.384, 96.402, 102.82, 109.06, 115.90, 121.40, 128.04, 133.69, 140.93, 147.30],
        "backgroundColor": "rgb(75, 112, 192)",
        "borderColor": "rgb(75, 112, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle (img2col)",
        "data": [140.13, 213.34, 304.83, 447.42, 590.65, 735.16, 868.25, 987.72, 1122.5, 1244.3, 1359.8, 1463.0, 1580.3, 1711.4, 1813.7, 1995.1, 2028.3, 2181.0, 2279.2, 2390.5, 2512.1, 2606.2, 2721.7],
        "backgroundColor": "rgba(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "animation": false,
    "responsive": true,
    "plugins": {
      "legend": {
        "position": "top"
      }
    },
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Time (ms)"
        }
      },
      "x": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "in channel"
        }
      }
    }
  }
}
```
:::

::: chartjs Conv2d f32 Performance (width = 256, height = 256, in channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": ["64", "96", "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", "576", "608", "640", "672", "704", "736", "768"],
    "datasets": [
        {
        "label": "Torch (mkldnn + direct)",
        "data": [18.023, 25.794, 33.049, 41.304, 48.821, 56.058, 63.912, 71.418, 78.524, 87.137, 94.035, 101.74, 108.46, 114.30, 119.54, 126.21, 135.99, 150.29, 155.49, 161.87, 172.70, 177.62, 186.70],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },{
        "label": "Hpt (direct)",
        "data": [10.846, 16.344, 21.937, 27.883, 33.260, 38.573, 44.207, 49.582, 56.990, 62.660, 68.631, 74.543, 80.089, 85.959, 91.965, 98.174, 103.83, 109.49, 115.91, 121.60, 126.98, 133.16, 138.88],
        "backgroundColor": "rgb(75, 112, 192)",
        "borderColor": "rgb(75, 112, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle (img2col)",
        "data": [268.58, 269.23, 307.83, 299.83, 326.94, 315.25, 423.30, 364.40, 431.93, 365.28, 458.44, 403.78, 457.27, 440.22, 623.03, 466.12, 555.98, 514.41, 711.03, 520.43, 641.99, 578.86, 858.27],
        "backgroundColor": "rgba(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "animation": false,
    "responsive": true,
    "plugins": {
      "legend": {
        "position": "top"
      }
    },
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Time (ms)"
        }
      },
      "x": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "out channel"
        }
      }
    }
  }
}
```
:::

::: chartjs Conv2d f32 Performance (in channel = 128, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": [4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
    "datasets": [
        {
        "label": "Torch (mkldnn + direct)",
        "data": [0.038644, 0.12771, 0.19875, 0.52712, 1.9615, 4.3826, 7.9169, 12.640, 18.465, 28.645, 33.039, 41.252, 51.089, 62.094, 73.428, 87.117, 99.958, 114.19, 131.18, 154.06, 166.08, 188.84, 211.78, 233.22, 252.82, 276.93, 300.65],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },{
        "label": "Hpt (direct)",
        "data": [0.16983, 0.30041, 0.41073, 0.73458, 1.4293, 3.0408, 5.6687, 8.5987, 12.981, 17.117, 22.631, 29.216, 36.362, 44.095, 50.709, 59.376, 69.827, 79.769, 90.664, 102.99, 115.86, 124.81, 138.10, 154.14, 169.82, 187.00, 208.88],
        "borderColor": "rgb(75, 112, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle (img2col)",
        "data": [0.042086, 0.10589, 0.41631, 1.9028, 13.855, 20.174, 56.339, 63.172, 147.46, 153.72, 295.42, 228.80, 412.75, 342.43, 569.62, 476.22, 820.45, 641.43, 1299.3, 824.72, 1418.8, 1019.5, 1508.7, 1257.9, 2149.3, 1527.0, 2903.7],
        "backgroundColor": "rgba(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 1
      }
    ]
  },
  "options": {
    "animation": false,
    "responsive": true,
    "plugins": {
      "legend": {
        "position": "top"
      }
    },
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Time (ms)"
        }
      },
      "x": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "width and height (width = height)"
        }
      }
    }
  }
}
```
:::

# Compilation config
```cargo
[profile.release]
opt-level = 3
incremental = true
debug = true
lto = "fat"
codegen-units = 1
```

# Running Threads
`10`

# Device specification
`CPU`: 12th Gen Intel(R) Core(TM) i5-12600K   3.69 GHz

`RAM`: G.SKILL Trident Z Royal Series (Intel XMP) DDR4 64GB

`System`: Windows 11 Pro 23H2