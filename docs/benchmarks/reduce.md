# Reduce Benchmark

::: chartjs Sum f32 Performance (size = 1024 * 2048 * 8)
```json
{
  "type": "bar",
  "data": {
    "labels": ["0", "1", "2", "(0, 1)", "(0, 2)", "(1, 2)", "(0, 1, 2)"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [2.3462, 1.8010, 3.4273, 4.2395, 8.8171, 1.8204, 1.7806],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [3.2746, 1.6347, 6.7277, 3.9968, 22.668, 1.6755, 1.6267],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [45.533, 45.716, 7.7589, 89.624, 89.743, 2.4209, 2.4785],
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
          "text": "Axes to reduce"
        }
      }
    }
  }
}
```
:::

::: chartjs Cuda Sum f32 Performance (size = 1024 * 1024 * 80)
```json
{
  "type": "bar",
  "data": {
    "labels": ["0", "1", "2", "(0, 1)", "(0, 2)", "(1, 2)", "(0, 1, 2)"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [401.86, 396.35, 427.3, 438.21, 455.26, 369.31, 367.62],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [373.82, 378.53, 393.38, 399.07, 405.73, 363.65, 391.65],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Candle",
        "data": [2210, 2930, 1110, 1220, 595.97, 589.44, 74610],
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
          "text": "Axes to reduce"
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

`GPU`: RTX 4090 24GB

`System`: Windows 11 Pro 23H2