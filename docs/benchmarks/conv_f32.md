::: chartjs Conv2d AVX2 f32 Performance (width = 256, height = 256, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": ["64", "96", "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", "576", "608", "640", "672", "704", "736", "768"],
    "datasets": [
        {
        "label": "Torch",
        "data": [18.706, 25.780, 32.960, 40.384, 48.180, 55.959, 65.984, 70.528, 77.862, 85.401, 92.009, 99.589, 105.72, 109.05, 114.86, 122.00, 129.14, 134.25, 141.98, 145.11, 154.19, 161.20, 166.91],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 2
      },
      {
        "label": "Hpt",
        "data": [10.822, 16.333, 21.808, 27.342, 33.014, 38.645, 44.315, 50.162, 55.687, 61.152, 66.840, 73.283, 78.407, 83.989, 89.650, 101.27, 102.64, 107.08, 112.94, 119.49, 124.58, 129.24, 135.68],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
        "borderWidth": 2
      },
      {
        "label": "Candle",
        "data": [140.13, 213.34, 304.83, 447.42, 590.65, 735.16, 868.25, 987.72, 1122.5, 1244.3, 1359.8, 1463.0, 1580.3, 1711.4, 1813.7, 1995.1, 2028.3, 2181.0, 2279.2, 2390.5, 2512.1, 2606.2, 2721.7],
        "backgroundColor": "rgba(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 2
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

::: chartjs Conv2d AVX512 f32 Performance (width = 256, height = 256, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": ["64", "96", "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", "576", "608", "640", "672", "704", "736", "768"],
    "datasets": [
        {
        "label": "mkldnn",
        "data": [3.69209, 4.9428, 6.7210, 7.65442, 9.3705, 10.5755, 12.863, 13.5580, 15.0237, 16.38044, 18.2367, 19.629, 21.0645, 22.74742, 24.9193, 25.9059, 28.421, 29.129, 31.207, 32.6317, 34.188, 35.201, 36.037],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 2
      },
      {
        "label": "Hpt",
        "data": [2.50, 3.620, 4.849, 5.965, 7.328, 8.623, 10.297, 11.746, 13.355, 15.002, 16.450, 18.0121, 19.144, 20.815, 22.837, 24.4733, 26.179, 27.393, 29.035, 30.389, 31.840, 33.435, 34.94],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
        "borderWidth": 2
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

::: chartjs Conv2d AVX2 f32 Performance (in channel = 128, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": [4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
    "datasets": [
        {
        "label": "mkldnn",
        "data": [0.038644, 0.12771, 0.19875, 0.52712, 1.9615, 4.3826, 7.9169, 12.640, 18.465, 28.645, 33.039, 41.252, 51.089, 62.094, 73.428, 87.117, 99.958, 114.19, 131.18, 154.06, 166.08, 188.84, 211.78, 233.22, 252.82, 276.93, 300.65],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 2
      },{
        "label": "Hpt",
        "data": [0.25723, 0.27866, 0.38936, 0.72296, 1.4765, 3.0682, 5.4992, 8.4730, 12.158, 16.717, 21.912, 27.975, 34.101, 41.608, 49.663, 58.880, 68.691, 79.496, 90.078, 101.34, 113.67, 126.73, 141.20, 155.80, 171.70, 188.71, 205.72],
        "backgroundColor": "rgb(0, 204, 255)",
        "borderColor": "rgb(0, 204, 255)",
        "borderWidth": 2
      },
      {
        "label": "Candle (img2col)",
        "data": [0.042086, 0.10589, 0.41631, 1.9028, 13.855, 20.174, 56.339, 63.172, 147.46, 153.72, 295.42, 228.80, 412.75, 342.43, 569.62, 476.22, 820.45, 641.43, 1299.3, 824.72, 1418.8, 1019.5, 1508.7, 1257.9, 2149.3, 1527.0, 2903.7],
        "backgroundColor": "rgba(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 2
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

::: chartjs Conv2d AVX512 f32 Performance (in channel = 128, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": [4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
    "datasets": [
        {
        "label": "mkldnn",
        "data": [0.050392, 0.0606, 0.1009, 0.18148, 0.5050, 1.0655, 1.5795, 2.996, 3.2918, 4.5697, 6.2755, 8.2038, 10.341, 12.7974, 15.4840, 19.895, 23.467, 27.033, 30.092, 36.504, 40.58394, 46.867, 54.756],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 2
      },{
        "label": "Hpt",
        "data": [0.01503, 0.08425, 0.178, 0.245, 0.517, 0.987, 1.596, 2.0255, 2.822, 3.677, 4.744, 6.1451, 7.326, 8.882,10.641, 12.742, 14.306, 16.553, 19.132, 22.414, 24.157, 27.4727, 29.038],
        "backgroundColor": "rgb(0, 204, 255)",
        "borderColor": "rgb(0, 204, 255)",
        "borderWidth": 2
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