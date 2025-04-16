
::: chartjs Conv2d f16 Performance (width = 256, height = 256, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": ["64", "96", "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", "576", "608", "640", "672", "704", "736", "768"],
    "datasets": [
      {
        "label": "Hpt (v0.1.2)",
        "data": [46.262, 69.652, 92.219, 118.24, 143.01, 169.41, 188.38, 214.79, 242.19, 267.62, 290.78, 300.47, 328.20, 359.88, 374.12, 399.66, 419.48, 443.11, 476.55, 491.03, 518.66, 540.00, 567.59],
        "backgroundColor": "rgb(25, 181, 243)",
        "borderColor": "rgb(25, 181, 243)",
        "borderWidth": 1
      },{
        "label": "Hpt (v0.1.3)",
        "data": [12.658, 18.227, 24.013, 29.687, 35.056, 40.969, 46.777, 52.131, 59.053, 63.639, 69.379, 75.276, 80.837, 86.738, 93.312, 97.810, 103.53, 109.88, 115.01, 121.29, 126.47, 132.12, 138.83],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
        "borderWidth": 1
      },
      {
        "label": "OnnxRuntime (img2col)",
        "data": [27.858, 38.516, 47.087, 54.080, 64.976, 71.196, 81.613, 91.683, 101.218, 113.884, 119.453, 131.129, 137.660, 146.9636, 153.921, 166.308, 174.351, 183.003, 193.423, 203.477, 207.67, 218.870, 226.678],
        "backgroundColor": "rgb(255, 94, 0)",
        "borderColor": "rgb(255, 94, 0)",
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

::: chartjs Conv2d f16 Performance (width = 256, height = 256, in channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": ["64", "96", "128", "160", "192", "224", "256", "288", "320", "352", "384", "416", "448", "480", "512", "544", "576", "608", "640", "672", "704", "736", "768"],
    "datasets": [{
        "label": "Hpt (v0.1.2)",
        "data": [40.216, 60.332, 81.317, 101.20, 121.86, 142.44, 163.231, 185.13, 204.92, 223.12, 244.36, 265.98, 285.85, 307.19, 328.57, 349.25, 367.74, 392.33, 410.31, 427.05, 452.29, 470.74, 487.04],
        "backgroundColor": "rgb(75, 112, 192)",
        "borderColor": "rgb(75, 112, 192)",
        "borderWidth": 1
      },{
        "label": "Hpt (v0.1.3)",
        "data": [11.792 , 17.819, 23.889, 30.363, 36.165, 42.455, 49.029, 55.250, 61.410, 68.653, 74.739, 82.292, 88.860, 94.940, 102.02, 106.70, 113.96, 119.90, 126.65, 133.08, 142.50, 151.48, 157.07],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
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

::: chartjs Conv2d f16 Performance (in channel = 128, out channel = 128, kernel size = 3)
```json
{
  "type": "line",
  "data": {
    "labels": [4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
    "datasets": [{
        "label": "Hpt (v0.1.2)",
        "data": [0.391, 0.690, 1.3783, 2.6697, 5.2947, 11.958, 20.665, 32.066, 45.711, 61.870, 81.878, 102.66, 126.15, 154.16, 181.76, 213.25, 246.15, 283.64, 321.32, 365.41, 408.30, 452.73, 503.49, 553.94, 608.68, 666.20, 722.03],
        "backgroundColor": "rgb(75, 112, 192)",
        "borderColor": "rgb(75, 112, 192)",
        "borderWidth": 1
      },{
        "label": "Hpt (v0.1.3)",
        "data": [0.192, 0.263, 0.517, 0.816, 1.5023, 3.2098, 5.5950, 9.0119, 13.261, 18.101, 23.933, 30.667, 38.166, 46.225, 54.760, 64.680, 75.642, 86.821, 99.483, 113.20, 126.37, 141.65, 157.88, 174.25, 191.36, 210.31, 226.55],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
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