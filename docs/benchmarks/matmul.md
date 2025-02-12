# Matmul Benchmark

::: chartjs Matmul f32 Performance (batch = 10, k = 4096)
```json
{
  "type": "line",
  "data": {
    "labels": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "datasets": [
        {
        "label": "Torch",
        "data": [0.010345, 0.077497, 0.090313, 0.24603, 0.81412, 3.4693, 12.333, 44.709, 171.05, 655.91],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Hpt",
        "data": [0.055021, 0.068729, 0.091027, 0.16254, 0.27781, 0.70462, 2.8303, 11.688, 35.024, 127.80],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle",
        "data": [0.0094133, 0.070120, 0.41147, 0.79394, 0.63238, 1.4742, 4.3279, 13.518, 37.915, 133.47],
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
          "text": "m, n (m = n)"
        }
      }
    }
  }
}
```
:::

::: chartjs Matmul f32 Performance (batch = 10, m = 4096, n = 4096)
```json
{
  "type": "line",
  "data": {
    "labels": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "datasets": [
        {
        "label": "Torch",
        "data": [91.793, 92.491, 94.433, 101.55, 142.60, 241.65, 393.87, 702.66, 1356.8, 2648],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "Hpt",
        "data": [60.608, 66.784, 66.798, 67.665, 70.486, 85.030, 108.37, 160.87, 274.78, 499.22],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Candle",
        "data": [124.78, 124.79, 124.67, 125.76, 124.34, 142.53, 170.97, 233.84, 379.85, 679.74],
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
          "text": "k"
        }
      }
    }
  }
}
```
:::