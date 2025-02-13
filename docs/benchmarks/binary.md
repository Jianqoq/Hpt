# Binary Benchmark

::: chartjs Binary f32 Performance (size = 100 * 100 * 100 * 100)
```json
{
  "type": "bar",
  "data": {
    "labels": ["add"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [37.061],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [48.140],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [58.089],
        "backgroundColor": "rgb(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 1
      },
      {
        "label": "Ndarray (par)",
        "data": [64.678],
        "backgroundColor": "rgb(32, 105, 241)",
        "borderColor": "rgb(32, 105, 241)",
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
      }
    }
  }
}
```
:::

::: chartjs Binary broadcast f32 Performance (lhs = 1 * 100 * 1 * 100, rhs = 100 * 1 * 100 * 1)
```json
{
  "type": "bar",
  "data": {
    "labels": ["add"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [23.842],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [25.216],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [330.83],
        "backgroundColor": "rgb(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 1
      },
      {
        "label": "Ndarray (par)",
        "data": [40.820],
        "backgroundColor": "rgb(32, 105, 241)",
        "borderColor": "rgb(32, 105, 241)",
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