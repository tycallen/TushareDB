<template>
  <div class="kline-chart-container">
    <div id="kline-chart"></div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue';
import * as echarts from 'echarts';
import axios from 'axios'; // 导入 Axios

let chart: echarts.ECharts | null = null;
const klineData = ref<any[]>([]);
const adjFactorData = ref<any[]>([]);

// 计算移动平均线
const calculateMA = (dayCount: number) => {
  const result: (number | '-')[] = [];
  for (let i = 0; i < klineData.value.length; i++) {
    if (i < dayCount) {
      result.push('-');
      continue;
    }
    let sum = 0;
    for (let j = 0; j < dayCount; j++) {
      sum += klineData.value[i - j].close;
    }
    result.push(+(sum / dayCount).toFixed(3));
  }
  return result;
};

const resizeChart = () => {
  if (chart) {
    chart.resize();
  }
};

// 定义获取 K 线数据和复权因子数据的函数
const fetchData = async () => {
  try {
    // 假设后端服务运行在 http://localhost:8000
    const baseUrl = 'http://localhost:8000';
    const tsCode = '000001.SZ'; // 示例股票代码
    const startDate = '20230101';
    const endDate = '20231231';

    // 获取 K 线数据
    const proBarResponse = await axios.get(`${baseUrl}/api/pro_bar`, {
      params: {
        ts_code: tsCode,
        start_date: startDate,
        end_date: endDate,
        adjfactor: true, // 请求复权数据
        fields: 'trade_date,open,high,low,close,vol,amount'
      }
    });
    klineData.value = proBarResponse.data;

    // 获取复权因子数据 (如果 pro_bar 接口已经处理了复权，这个接口可能不需要单独调用，这里作为示例保留)
    const adjFactorResponse = await axios.get(`${baseUrl}/api/adj_factor`, {
      params: {
        ts_code: tsCode,
        start_date: startDate,
        end_date: endDate,
        fields: 'trade_date,adj_factor'
      }
    });
    adjFactorData.value = adjFactorResponse.data;

    console.log('K-line Data:', klineData.value);
    console.log('Adj Factor Data:', adjFactorData.value);

    // 数据获取成功后，更新图表
    updateChart();

  } catch (error) {
    console.error('Error fetching data:', error);
  }
};

// 更新 ECharts 图表的函数
const updateChart = () => {
  if (!chart || klineData.value.length === 0) {
    return;
  }

  // 准备 ECharts 需要的数据格式
  const dates = klineData.value.map((item: any) => item.trade_date);
  const values = klineData.value.map((item: any) => [
    item.open,
    item.close,
    item.low,
    item.high
  ]);
  const volumes = klineData.value.map((item: any) => item.vol);

  // 计算移动平均线
  const ma5 = calculateMA(5);
  const ma10 = calculateMA(10);
  const ma20 = calculateMA(20);

  chart.setOption({
    xAxis: [
      {
        data: dates
      },
      {
        data: dates
      }
    ],
    series: [
      {
        name: 'Candlestick',
        type: 'candlestick',
        data: values,
        itemStyle: {
          color: '#ec0000',
          color0: '#00da3c',
          borderColor: '#ec0000',
          borderColor0: '#00da3c'
        },
        markPoint: {
          label: {
            formatter: function (param: any) {
              return param != null ? Math.round(param.value) + '' : '';
            }
          },
          data: [
            {
              name: 'highest value',
              type: 'max',
              valueDim: 'highest'
            },
            {
              name: 'lowest value',
              type: 'min',
              valueDim: 'lowest'
            },
            {
              name: 'average value on close',
              type: 'average',
              valueDim: 'close'
            }
          ],
          tooltip: {
            formatter: function (param: any) {
              return param.name + '<br>' + (param.data.value || '');
            }
          }
        },
        markLine: {
          label: {
            formatter: function (param: any) {
              return param.value + '';
            }
          },
          data: [
            {
              name: 'min line on close',
              type: 'min',
              valueDim: 'close'
            },
            {
              name: 'max line on close',
              type: 'max',
              valueDim: 'close'
            },
            {
              name: 'average line on close',
              type: 'average',
              valueDim: 'close'
            }
          ]
        }
      },
      {
        name: 'MA5',
        type: 'line',
        data: ma5,
        smooth: true,
        lineStyle: {
          opacity: 0.5
        }
      },
      {
        name: 'MA10',
        type: 'line',
        data: ma10,
        smooth: true,
        lineStyle: {
          opacity: 0.5
        }
      },
      {
        name: 'MA20',
        type: 'line',
        data: ma20,
        smooth: true,
        lineStyle: {
          opacity: 0.5
        }
      },
      {
        name: 'Volume',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: volumes,
        itemStyle: {
          color: (params: any) => {
            const dataIndex = params.dataIndex;
            const open = klineData.value[dataIndex].open;
            const close = klineData.value[dataIndex].close;
            return close > open ? '#ec0000' : '#00da3c';
          }
        }
      }
    ],
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      },
      formatter: function (params: any) {
        let res = params[0].name + '<br/>';
        for (let i = 0; i < params.length; i++) {
          if (params[i].seriesType === 'candlestick') {
            res += '开盘: ' + params[i].data[0] + '<br/>';
            res += '收盘: ' + params[i].data[1] + '<br/>';
            res += '最低: ' + params[i].data[2] + '<br/>';
            res += '最高: ' + params[i].data[3] + '<br/>';
          } else if (params[i].seriesName === 'Volume') {
            res += '成交量: ' + params[i].value + '<br/>';
          } else {
            res += params[i].seriesName + ': ' + params[i].value + '<br/>';
          }
        }
        return res;
      }
    },
  });
};


onMounted(() => {
  const chartDom = document.getElementById('kline-chart');
  if (chartDom) {
    chart = echarts.init(chartDom);
    // 初始设置，确保图表容器有尺寸
    chart.setOption({
      title: {
        text: 'K线图'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['Candlestick', 'MA5', 'MA10', 'MA20']
      },
      grid: [
        {
          left: '10%',
          right: '8%',
          height: '50%'
        },
        {
          left: '10%',
          right: '8%',
          top: '63%',
          height: '16%'
        }
      ],
      xAxis: [
        {
          type: 'category',
          data: [],
          scale: true,
          boundaryGap: false,
          axisLine: { onZero: false },
          splitLine: { show: false },
          splitNumber: 20,
          min: 'dataMin',
          max: 'dataMax'
        },
        {
          type: 'category',
          gridIndex: 1,
          data: [],
          scale: true,
          boundaryGap: false,
          axisLine: { onZero: false },
          axisTick: { show: false },
          splitLine: { show: false },
          axisLabel: { show: false },
          min: 'dataMin',
          max: 'dataMax'
        }
      ],
      yAxis: [
        {
          scale: true,
          splitArea: {
            show: true
          }
        },
        {
          gridIndex: 1,
          scale: true,
          splitNumber: 2,
          axisLabel: { show: false },
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { show: false }
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: 0,
          end: 100
        },
        {
          show: true,
          xAxisIndex: [0, 1],
          type: 'slider',
          top: '85%',
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: 'Candlestick',
          type: 'candlestick',
          data: []
        },
        {
          name: 'MA5',
          type: 'line',
          data: [],
          smooth: true,
          lineStyle: {
            opacity: 0.5
          }
        },
        {
          name: 'MA10',
          type: 'line',
          data: [],
          smooth: true,
          lineStyle: {
            opacity: 0.5
          }
        },
        {
          name: 'MA20',
          type: 'line',
          data: [],
          smooth: true,
          lineStyle: {
            opacity: 0.5
          }
        },
        {
          name: 'Volume',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: []
        }
      ]
    });
    fetchData(); // 在组件挂载后获取数据
    window.addEventListener('resize', resizeChart); // 添加 resize 事件监听
  }
});

onUnmounted(() => {
  if (chart) {
    chart.dispose();
    chart = null;
  }
  window.removeEventListener('resize', resizeChart); // 移除 resize 事件监听
});
</script>

<style scoped>
.kline-chart-container {
  width: 100%;
  height: 100%; /* 修改：使其填充父容器的高度 */
  display: flex;
  justify-content: center;
  align-items: center;
}

#kline-chart {
  width: 100%;
  height: 100%; /* 填充父容器的高度 */
}
</style>