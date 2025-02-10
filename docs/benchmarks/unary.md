::: chartjs Unary f32 Performance (size = 1024 * 2048 * 8)
```json
{
  "type": "bar",
  "data": {
    "labels": ["sin", "exp", "log", "sqrt", "tanh", "gelu"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [4.5152, 4.3741, 4.6562, 4.4349, 6.1527, 5.1197],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [5.7344, 5.7475, 40.439, 37.416, 5.9281, 10.076],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [9.7761, 9.3340, 127.48, 124.98, 9.4629, 25.956],
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
      }
    }
  }
}
```
:::

::: chartjs Unary f32 Performance (size = 4096 * 2048 * 8)
```json
{
  "type": "bar",
  "data": {
    "labels": ["sin", "exp", "log", "sqrt", "tanh", "gelu"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [18.189, 17.594, 19.683, 18.031, 25.582, 20.822],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [28.158, 27.325, 165.71, 149.80, 28.290, 19.369],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [37.246, 39.440, 503.73, 491.34, 37.443, 105.69],
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
      }
    }
  }
}
```
:::

# Error precision (lower is better)
- `Hpt`: 10 ulps

- `Torch`: 10 ulps

- `Candle (mkl)`: 1 ulps

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