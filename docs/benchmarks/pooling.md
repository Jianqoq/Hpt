# Pooling Benchmark

::: chartjs MaxPool f32 Performance (width = 256, height = 256, kernel size = 4)
```json
{
  "type": "line",
  "data": {
    "labels": [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
    "datasets": [
        {
        "label": "Torch (mkldnn)",
        "data": [32.386, 48.811, 64.403, 80.544, 95.353, 111.11, 127.24, 143.18, 158.83, 171.98, 188.59, 203.81, 219.72, 234.57, 250.49, 266.12, 281.03, 295.86, 310.05, 326.17, 343.78, 355.07, 371.32],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Hpt",
        "data": [1.2438, 2.1882, 3.0853, 3.9226, 4.9102, 5.8994, 6.7942, 8.0124, 9.1809, 10.387, 11.906, 13.354, 14.707, 16.338, 17.670, 18.888, 20.418, 21.174, 23.163, 24.425, 25.770, 26.962, 28.641],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle",
        "data": [2.2266, 3.4538, 4.5851, 5.8429, 6.9738, 8.6044, 9.3018, 10.566, 12.014, 12.930, 13.790, 15.465, 16.715, 18.186, 18.985, 21.397, 21.815, 23.113, 24.114, 25.409, 25.794, 27.847, 28.031],
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