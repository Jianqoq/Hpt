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
        "data": [15.823, 17.308, 18.801, 20.1400,21.4282, 22.6700, 24.4327, 25.76,27.170,28.622,30.265, 31.017, 32.930, 33.44342, 34.9395
        ],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 2
      },
      {
        "label": "OnnxRuntime",
        "data": [12.433, 13.732, 15.20, 16.603, 18.24, 19.219, 21.17, 22.762, 24.377, 26.080, 28.351, 29.374, 31.492, 32.643, 34.086],
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