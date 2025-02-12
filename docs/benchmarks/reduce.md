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
`CPU`: i5-12600k

`RAM`: G.SKILL Trident Z Royal Series (Intel XMP) DDR4 64GB