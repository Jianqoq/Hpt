# FFTN Complex32 Benchmark

::: chartjs
```json
{
  "type": "bar",
  "data": {
    "labels": ["[16, 16, 16]", "[32, 32, 32]", "[64, 64, 64]"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [0.18240, 0.22768, 0.78123],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [0.0060414, 0.040165, 0.43672],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
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
          "text": "Shape"
        }
      }
    }
  }
}
```
:::

::: chartjs
```json
{
  "type": "bar",
  "data": {
    "labels": ["[128, 128, 128]", "[256, 256, 256]"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [5.2680, 48.222],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [6.6137, 68.117],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
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
          "text": "Shape"
        }
      }
    }
  }
}
```
:::

::: chartjs
```json
{
  "type": "bar",
  "data": {
    "labels": ["[512, 512, 512]"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [770.23],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch",
        "data": [1098.3],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
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
          "text": "Shape"
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