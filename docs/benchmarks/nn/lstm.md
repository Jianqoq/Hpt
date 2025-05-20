::: chartjs LSTM Neon
```json
{
  "type": "line",
  "data": {
    "labels": [64, 128, 256 ],
    "datasets": [
      {
        "label": "Hpt",
        "data": [20.218687, 41.736266, 96.670106],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "OnnxRuntime",
        "data": [13.213, 31.098, 82.146],
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
          "text": "hidden size"
        }
      }
    }
  }
}
```
:::