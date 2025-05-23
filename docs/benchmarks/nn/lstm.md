::: chartjs LSTM f32 (batch = 128, seq_len = 256, input_size = 128)
```json
{
  "type": "line",
  "data": {
    "labels": [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512],
    "datasets": [
      {
        "label": "Hpt",
        "data": [23.940, 29.9874, 36.264, 47.50671, 57.4132, 68.16517, 80.9325, 94.3421,
        109.398, 126.992, 146.0649, 164.7794, 185.2949, 209.4334, 233.9380
        ],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 2
      },
      {
        "label": "OnnxRuntime",
        "data": [13.213, 21.352, 31.098, 41.487, 53.716, 66.741, 82.146, 100.308, 117.759, 135.352, 159.872, 
        183.950, 210.132, 241.228, 276.783],
        "backgroundColor": "rgb(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 2
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

::: chartjs LSTM f32 (batch = 128, seq_len = 256, hidden_size = 128)
```json
{
  "type": "line",
  "data": {
    "labels": [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512],
    "datasets": [
      {
        "label": "Hpt",
        "data": [23.940, 29.9874, 36.264, 47.50671, 57.4132, 68.16517, 80.9325, 94.3421,
        109.398, 126.992, 146.0649, 164.7794, 185.2949, 209.4334, 233.9380
        ],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 2
      },
      {
        "label": "OnnxRuntime",
        "data": [13.213, 21.352, 31.098, 41.487, 53.716, 66.741, 82.146, 100.308, 117.759, 135.352, 159.872, 
        183.950, 210.132, 241.228, 276.783],
        "backgroundColor": "rgb(255, 206, 86)",
        "borderColor": "rgb(255, 206, 86)",
        "borderWidth": 2
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