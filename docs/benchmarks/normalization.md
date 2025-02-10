::: chartjs Softmax f32 Performance (size = 128 * 128 * 128)
```json
{
  "type": "bar",
  "data": {
    "labels": ["0", "1", "2"],
    "datasets": [
      {
        "label": "Torch",
        "data": [0.76168, 0.42298, 0.44949],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Hpt",
        "data": [1.1718, 1.1321, 0.49570],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [17.392, 21.962, 6.6240],
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

::: chartjs Softmax f32 Performance (size = 256 * 256 * 256)
```json
{
  "type": "bar",
  "data": {
    "labels": ["0", "1", "2"],
    "datasets": [
      {
        "label": "Torch",
        "data": [16.263, 7.3493, 6.2443],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Hpt",
        "data": [13.305, 9.1866, 6.2031],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [249.70, 171.54, 58.431],
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

::: chartjs Softmax f32 Performance (size = 512 * 512 * 512)
```json
{
  "type": "bar",
  "data": {
    "labels": ["0", "1", "2"],
    "datasets": [
      {
        "label": "Torch",
        "data": [235.78, 85.570, 61.285],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Hpt",
        "data": [125.77, 83.385, 48.184],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle (mkl)",
        "data": [2820.6, 1397.3, 486.96],
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