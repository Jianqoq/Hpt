::: chartjs Small LSTM
```json
{
  "type": "line",
  "data": {
    "labels": ["(128, 64, 1, 128)", "(128, 64, 1, 128)", "(128, 64, 1, 128)"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [20.218687, 41.736266, 96.670106],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch(mkldnn)",
        "data": [10.270547866821289, 10.283946990966797, 13.933086395263672],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
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
          "text": "(inp_size, hidden, num_layer, output_size)"
        }
      }
    }
  }
}
```
:::

::: chartjs Mideum LSTM
```json
{
  "type": "line",
  "data": {
    "labels": ["(inp_size=200, hidden=128, num_layer=3, output_size=256)", "(200, 256, 3, 128)", "(200, 512, 3, 128)"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [11.810137, 18.928571, 36.724731],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch(mkldnn)",
        "data": [31.58514499664307, 36.01236343383789, 66.0283088684082],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "OnnxRuntime",
        "data": [5.274081230163574, 9.469199180603027, 22.227072715759277],
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
          "text": "batch size"
        }
      }
    }
  }
}
```
:::

::: chartjs Large LSTM
```json
{
  "type": "line",
  "data": {
    "labels": ["(inp_size=512, hidden=512, num_layer=5, output_size=512)", "(512, 768, 5, 512)", "(512, 1024, 5, 512)"],
    "datasets": [
      {
        "label": "Hpt",
        "data": [62.282539, 152.660275, 287.603293],
        "backgroundColor": "rgb(75, 192, 192)",
        "borderColor": "rgb(75, 192, 192)",
        "borderWidth": 1
      },
      {
        "label": "Torch(mkldnn)",
        "data": [111.32850646972656, 256.22849464416504, 376.4298915863037],
        "backgroundColor": "rgb(255, 99, 133)",
        "borderColor": "rgb(255, 99, 132)",
        "borderWidth": 1
      },
      {
        "label": "OnnxRuntime",
        "data": [38.37249279022217, 87.07950115203857, 179.89604473114014],
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
          "text": "batch size"
        }
      }
    }
  }
}
```
:::