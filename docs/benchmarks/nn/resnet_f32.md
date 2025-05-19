# Resnet34(f32) Neon Benchmark

::: chartjs Input (batch = 1, in channel = 3)
```json
{
  "type": "line",
  "data": {
    "labels": [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
    "datasets": [
      {
        "label": "tract",
        "data": [4.430394, 6.584067, 7.759865, 12.59866, 16.2582, 22.915, 23.8278, 36.6217, 41.7674, 52.5317, 55.6117, 72.991, 79.15411, 97.9825, 95.8596, 127.8529, 132.281, 160.434, 161.4075, 196.234, 202.17, 232.0213, 218.0608],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },{
        "label": "Hpt",
        "data": [10.953, 13.60794, 16.266149, 21.270495, 25.322886, 31.477429, 34.5868, 43.030, 48.924, 56.716, 65.345, 75.445, 82.722, 97.74, 106.948, 122.057, 136.368, 149.844, 158.28, 169.174, 181.167, 201.970, 208.33, 228.77],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
        "borderWidth": 1
      },
      {
        "label": "OnnxRuntime",
        "data": [4.4556, 7.071, 9.453, 14.374, 19.52, 27.41, 29.82, 41.378, 47.911, 56.100, 65.52, 78.07, 90.162, 103.40, 113.327, 131.593, 145.342, 160.650, 177.760, 197.503, 216.1836, 232.2501, 254.5794],
        "backgroundColor": "rgb(255, 206, 86)",
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
          "text": "width and height"
        }
      }
    }
  }
}
```
:::

::: chartjs Input (batch = 5, in channel = 3)
```json
{
  "type": "line",
  "data": {
    "labels": [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024],
    "datasets": [
      {
        "label": "Hpt",
        "data": [16.747468, 19.869115, 22.254, 27.211392, 31.267589, 36.817232, 40.973727, 49.235898, 56.790511, 62.4614, 69.7877, 79.79704, 89.030461, 101.074937, 109.957399,
        125.743765, 136.728914, 152.670072, 167.788094, 187.387096, 197.507286, 216.234451, 232.46538],
        "backgroundColor": "rgb(116, 211, 28)",
        "borderColor": "rgb(116, 211, 28)",
        "borderWidth": 1
      },
      {
        "label": "OnnxRuntime",
        "data": [12.508487701416016, 14.993786811828613, 18.39921474456787, 22.306108474731445, 27.615714073181152, 34.2897891998291, 37.95499801635742, 49.868178367614746, 55.8765172958374, 64.32678699493408, 73.19142818450928, 85.51030158996582, 98.0064868927002, 112.63859272003174, 123.0414867401123, 140.1526, 154.6250104, 171.4838, 186.414384, 206.3034, 227.565908, 245.0677, 260.855],
        "backgroundColor": "rgb(255, 206, 86)",
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
          "text": "width and height"
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