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
        "label": "Hpt (v0.1.2)",
        "data": [11.907, 17.747, 23.966, 30.349, 36.190, 42.790, 49.520, 58.015, 65.420, 72.911, 79.606, 87.289, 94.646, 104.40, 112.79, 122.16, 127.39, 133.96, 149.39, 154.46, 163.72, 173.04, 182.77],
        "backgroundColor": "rgb(25, 181, 243)",
        "borderColor": "rgb(25, 181, 243)",
        "borderWidth": 1
      },
      {
        "label": "Hpt (v0.1.3)",
        "data": [10.822, 16.333, 21.808, 27.342, 33.014, 38.645, 44.315, 50.162, 55.687, 61.152, 66.840, 73.283, 78.407, 83.989, 89.650, 101.27, 102.64, 107.08, 112.94, 119.49, 124.58, 129.24, 135.68],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
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
        "label": "Hpt (v0.1.2)",
        "data": [12.602, 18.278, 23.997, 29.788, 35.310, 41.075, 47.761, 54.036, 60.597, 66.913, 72.801, 78.695, 84.515, 90.725, 97.012, 103.95, 110.08, 115.81, 123.06, 129.11, 135.18, 141.17, 147.51],
        "backgroundColor": "rgb(75, 112, 192)",
        "borderColor": "rgb(75, 112, 192)",
        "borderWidth": 1
      },{
        "label": "Hpt (v0.1.3)",
        "data": [10.744, 16.188, 21.850, 27.067, 32.813, 37.941, 43.470, 49.366, 54.768, 60.613, 66.822, 71.855, 77.842, 83.991, 90.347, 95.500, 103.16, 107.05, 118.46, 120.39, 125.83, 133.02, 141.34],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
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
        "backgroundColor": "rgb(75, 112, 192)",
        "borderColor": "rgb(75, 112, 192)",
        "borderWidth": 1
      },{
        "label": "Hpt (v0.1.3)",
        "data": [0.25723, 0.27866, 0.38936, 0.72296, 1.4765, 3.0682, 5.4992, 8.4730, 12.158, 16.717, 21.912, 27.975, 34.101, 41.608, 49.663, 58.880, 68.691, 79.496, 90.078, 101.34, 113.67, 126.73, 141.20, 155.80, 171.70, 188.71, 205.72],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
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