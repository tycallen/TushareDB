<template>
  <div class="stock-detail-view">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="box-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span>股票基本信息</span>
            </div>
          </template>
          <div v-if="stockBasicInfo" class="stock-basic-info">
            <el-descriptions :column="3" border>
              <el-descriptions-item label="股票代码">{{ stockBasicInfo.ts_code }}</el-descriptions-item>
              <el-descriptions-item label="股票名称">{{ stockBasicInfo.name }}</el-descriptions-item>
              <el-descriptions-item label="所在市场">{{ stockBasicInfo.market }}</el-descriptions-item>
              <el-descriptions-item label="地域">{{ stockBasicInfo.area }}</el-descriptions-item>
              <el-descriptions-item label="所属行业">{{ stockBasicInfo.industry }}</el-descriptions-item>
              <el-descriptions-item label="上市日期">{{ stockBasicInfo.list_date }}</el-descriptions-item>
            </el-descriptions>
          </div>
          <div v-else>
            <el-empty description="暂无股票基本信息"></el-empty>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="24">
        <el-card class="box-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span>实时指标卡片</span>
            </div>
          </template>
          <div v-if="dailyBasicInfo" class="daily-basic-info">
            <el-row :gutter="20">
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">当日收盘价</div>
                  <div class="metric-value">{{ dailyBasicInfo.close }}</div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">换手率 (%)</div>
                  <div class="metric-value">{{ (dailyBasicInfo.turn_over * 100).toFixed(2) }}</div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">量比</div>
                  <div class="metric-value">{{ dailyBasicInfo.vol_ratio ? dailyBasicInfo.vol_ratio.toFixed(2) : 'N/A' }}</div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">市盈率 (PE)</div>
                  <div class="metric-value">{{ dailyBasicInfo.pe ? dailyBasicInfo.pe.toFixed(2) : 'N/A' }}</div>
                </el-card>
              </el-col>
            </el-row>
            <el-row :gutter="20" style="margin-top: 20px;">
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">市净率 (PB)</div>
                  <div class="metric-value">{{ dailyBasicInfo.pb ? dailyBasicInfo.pb.toFixed(2) : 'N/A' }}</div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">总市值 (亿)</div>
                  <div class="metric-value">{{ (dailyBasicInfo.total_mv / 10000).toFixed(2) }}</div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">流通市值 (亿)</div>
                  <div class="metric-value">{{ (dailyBasicInfo.circ_mv / 10000).toFixed(2) }}</div>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="metric-card">
                  <div class="metric-title">自由流通市值 (亿)</div>
                  <div class="metric-value">{{ dailyBasicInfo.free_share_mv ? (dailyBasicInfo.free_share_mv / 10000).toFixed(2) : 'N/A' }}</div>
                </el-card>
              </el-col>
            </el-row>
          </div>
          <div v-else>
            <el-empty description="暂无实时指标数据"></el-empty>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="24">
        <el-card class="box-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span>K 线图</span>
            </div>
          </template>
          <div id="k-line-chart" style="width: 100%; height: 400px;"></div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="24">
        <el-card class="box-card" shadow="hover">
          <template #header>
            <div class="card-header">
              <span>详细数据</span>
            </div>
          </template>
          <el-tabs v-model="activeTab" type="border-card">
            <el-tab-pane label="公司资料" name="companyInfo">
              <div v-if="stockCompanyInfo" class="company-info">
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">公司名称:</span>
                      <span class="info-value">{{ stockCompanyInfo.com_name }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">英文名称:</span>
                      <span class="info-value">{{ stockCompanyInfo.en_name }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">注册资本(万元):</span>
                      <span class="info-value">{{ stockCompanyInfo.reg_capital }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">省份:</span>
                      <span class="info-value">{{ stockCompanyInfo.province }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">城市:</span>
                      <span class="info-value">{{ stockCompanyInfo.city }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">公司网址:</span>
                      <span class="info-value"><a :href="stockCompanyInfo.website" target="_blank">{{ stockCompanyInfo.website }}</a></span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">电子邮箱:</span>
                      <span class="info-value">{{ stockCompanyInfo.email }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">员工人数:</span>
                      <span class="info-value">{{ stockCompanyInfo.employees }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="24">
                    <div class="info-item">
                      <span class="info-label">主要业务:</span>
                      <span class="info-value">{{ stockCompanyInfo.main_business }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="24">
                    <div class="info-item">
                      <span class="info-label">经营范围:</span>
                      <span class="info-value">{{ stockCompanyInfo.business_scope }}</span>
                    </div>
                  </el-col>
                </el-row>
              </div>
              <div v-else>
                <el-empty description="暂无公司资料"></el-empty>
              </div>
            </el-tab-pane>
            <el-tab-pane label="财务指标" name="finaIndicator">
              <div v-if="finaIndicatorInfo" class="fina-indicator-info">
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">报告期:</span>
                      <span class="info-value">{{ finaIndicatorInfo.end_date }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">基本每股收益:</span>
                      <span class="info-value">{{ finaIndicatorInfo.eps }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">扣除非经常性损益后的基本每股收益:</span>
                      <span class="info-value">{{ finaIndicatorInfo.dedu_eps }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">每股营业总收入:</span>
                      <span class="info-value">{{ finaIndicatorInfo.eps_roe }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">每股净资产:</span>
                      <span class="info-value">{{ finaIndicatorInfo.bps }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">每股未分配利润:</span>
                      <span class="info-value">{{ finaIndicatorInfo.undist_profit_per_share }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">每股经营活动现金流量净额:</span>
                      <span class="info-value">{{ finaIndicatorInfo.op_cashflow_per_share }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">净资产收益率:</span>
                      <span class="info-value">{{ (finaIndicatorInfo.roe * 100).toFixed(2) + '%' }}</span>
                    </div>
                  </el-col>
                </el-row>
                <el-row :gutter="20">
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">销售毛利率:</span>
                      <span class="info-value">{{ (finaIndicatorInfo.gross_margin * 100).toFixed(2) + '%' }}</span>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="info-item">
                      <span class="info-label">销售净利率:</span>
                      <span class="info-value">{{ (finaIndicatorInfo.net_profit_ratio * 100).toFixed(2) + '%' }}</span>
                    </div>
                  </el-col>
                </el-row>
              </div>
              <div v-else>
                <el-empty description="暂无财务指标数据"></el-empty>
              </div>
            </el-tab-pane>
            <el-tab-pane label="筹码分布" name="cyqChips">
              <div v-if="cyqChipsInfo" class="cyq-chips-info">
                <el-descriptions :column="2" border>
                  <el-descriptions-item label="交易日期">{{ cyqChipsInfo.trade_date }}</el-descriptions-item>
                  <el-descriptions-item label="收盘价">{{ cyqChipsInfo.close }}</el-descriptions-item>
                  <el-descriptions-item label="平均成本">{{ cyqChipsInfo.avg_cost }}</el-descriptions-item>
                  <el-descriptions-item label="获利比例 (%)">{{ (cyqChipsInfo.profit_ratio * 100).toFixed(2) }}</el-descriptions-item>
                  <el-descriptions-item label="90%筹码集中度">{{ cyqChipsInfo.chips_90 }}</el-descriptions-item>
                  <el-descriptions-item label="70%筹码集中度">{{ cyqChipsInfo.chips_70 }}</el-descriptions-item>
                  <el-descriptions-item label="50%筹码集中度">{{ cyqChipsInfo.chips_50 }}</el-descriptions-item>
                  <el-descriptions-item label="20%筹码集中度">{{ cyqChipsInfo.chips_20 }}</el-descriptions-item>
                </el-descriptions>
              </div>
              <div v-else>
                <el-empty description="暂无筹码分布数据"></el-empty>
              </div>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { useRoute } from 'vue-router';
import * as echarts from 'echarts';
import axios from 'axios';

const route = useRoute();
const ts_code = ref(route.params.ts_code);
const stockBasicInfo = ref(null);
const dailyBasicInfo = ref(null);
const stockCompanyInfo = ref(null);
const finaIndicatorInfo = ref(null);
const cyqChipsInfo = ref(null); // 新增筹码分布数据
const loadingFinaIndicator = ref(false);
const activeTab = ref('companyInfo'); // 默认激活公司资料tab
const selectedPeriod = ref('20231231'); // 财务指标查询的默认报告期

// 获取股票基本信息
const fetchStockBasicInfo = async () => {
  try {
    const response = await axios.get(`http://localhost:8000/api/stock_basic`, {
      params: { ts_code: ts_code.value }
    });
    if (response.data && response.data.length > 0) {
      stockBasicInfo.value = response.data[0];
    } else {
      stockBasicInfo.value = null;
    }
  } catch (error) {
    console.error('Error fetching stock basic info:', error);
    stockBasicInfo.value = null;
  }
};

// 获取实时指标数据
const fetchDailyBasicInfo = async () => {
  try {
    const response = await axios.get(`http://localhost:8000/api/daily_basic`, {
      params: { ts_code: ts_code.value, trade_date: getTodayDate() } // 假设获取最新交易日数据
    });
    if (response.data && response.data.length > 0) {
      dailyBasicInfo.value = response.data[0];
    } else {
      dailyBasicInfo.value = null;
    }
  } catch (error) {
    console.error('Error fetching daily basic info:', error);
    dailyBasicInfo.value = null;
  }
};

// 获取公司资料
const fetchStockCompanyInfo = async () => {
  try {
    const response = await axios.get(`http://localhost:8000/api/stock_company`, {
      params: { ts_code: ts_code.value }
    });
    if (response.data && response.data.length > 0) {
      stockCompanyInfo.value = response.data[0];
    } else {
      stockCompanyInfo.value = null;
    }
  } catch (error) {
    console.error('Error fetching stock company info:', error);
    stockCompanyInfo.value = null;
  }
};

// 获取财务指标数据
const fetchFinaIndicatorInfo = async () => {
  loadingFinaIndicator.value = true;
  try {
    const response = await axios.get(`http://localhost:8000/api/fina_indicator_vip`, {
      params: { ts_code: ts_code.value, period: selectedPeriod.value }
    });
    if (response.data && response.data.length > 0) {
      finaIndicatorInfo.value = response.data[0];
    } else {
      finaIndicatorInfo.value = null;
    }
  } catch (error) {
    console.error('Error fetching financial indicator info:', error);
    finaIndicatorInfo.value = null;
  } finally {
    loadingFinaIndicator.value = false;
  }
};

// 获取筹码分布数据
const fetchCyqChipsInfo = async () => {
  try {
    const response = await axios.get(`http://localhost:8000/api/cyq_chips`, {
      params: { ts_code: ts_code.value, trade_date: '20250901' } // 假设获取最新交易日数据
    });
    if (response.data && response.data.length > 0) {
      cyqChipsInfo.value = response.data[0];
    } else {
      cyqChipsInfo.value = null;
    }
  } catch (error) {
    console.error('Error fetching CYQ chips info:', error);
    cyqChipsInfo.value = null;
  }
};

// 获取 K 线图数据并渲染
const fetchKLineDataAndRenderChart = async () => {
  try {
    const response = await axios.get(`http://localhost:8000/api/pro_bar`, {
      params: {
        ts_code: ts_code.value,
        start_date: '20230101', // 示例：获取一年数据
        end_date: getTodayDate(),
        freq: 'D',
        adjfactor: true
      }
    });
    if (response.data && response.data.length > 0) {
      renderKLineChart(response.data);
    } else {
      renderKLineChart([]); // 传入空数据以清空图表
    }
  } catch (error) {
    console.error('Error fetching K-line data:', error);
    renderKLineChart([]); // 发生错误时清空图表
  }
};

const renderKLineChart = (data) => {
  const chartDom = document.getElementById('k-line-chart');
  const myChart = echarts.init(chartDom);

  const dates = data.map(item => item.trade_date);
  const values = data.map(item => [
    item.open,
    item.close,
    item.low,
    item.high
  ]);

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    grid: {
      left: '10%',
      right: '10%',
      bottom: '15%'
    },
    xAxis: {
      type: 'category',
      data: dates,
      scale: true,
      boundaryGap: false,
      axisLine: { onZero: false },
      splitLine: { show: false },
      splitNumber: 20,
      min: 'dataMin',
      max: 'dataMax'
    },
    yAxis: {
      scale: true,
      splitArea: {
        show: true
      }
    },
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: [0, 1],
        start: 80,
        end: 100
      },
      {
        show: true,
        xAxisIndex: [0, 1],
        type: 'slider',
        bottom: '10%',
        start: 80,
        end: 100
      }
    ],
    series: [
      {
        name: 'K-line',
        type: 'candlestick',
        data: values,
        itemStyle: {
          color: '#ec0000',
          color0: '#00da3c',
          borderColor: '#ec0000',
          borderColor0: '#00da3c'
        }
      }
    ]
  };

  myChart.setOption(option);
};

// 获取今天的日期，格式为 YYYYMMDD
const getTodayDate = () => {
  const date = new Date();
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, '0');
  const day = date.getDate().toString().padStart(2, '0');
  return `${year}${month}${day}`;
};

onMounted(() => {
  if (ts_code.value) {
    fetchStockBasicInfo();
    fetchDailyBasicInfo();
    fetchKLineDataAndRenderChart();
    fetchStockCompanyInfo();
    fetchFinaIndicatorInfo();
    fetchCyqChipsInfo(); // 在组件挂载时获取筹码分布数据
  }
});

watch(() => route.params.ts_code, (newTsCode) => {
  if (newTsCode) {
    ts_code.value = newTsCode;
    fetchStockBasicInfo();
    fetchDailyBasicInfo();
    fetchKLineDataAndRenderChart();
    fetchStockCompanyInfo();
    fetchFinaIndicatorInfo();
    fetchCyqChipsInfo(); // 在 ts_code 变化时获取筹码分布数据
  }
});
</script>

<style scoped>
.stock-detail-view {
  padding: 20px;
}

.card-header {
  font-size: 18px;
  font-weight: bold;
}

.stock-basic-info,
.daily-basic-info,
.company-info,
.fina-indicator-info,
.cyq-chips-info {
  font-size: 14px;
  line-height: 1.8;
}

.metric-card {
  text-align: center;
}

.metric-title {
  font-size: 14px;
  color: #909399;
  margin-bottom: 5px;
}

.metric-value {
  font-size: 20px;
  font-weight: bold;
  color: #303133;
}

.info-item {
  display: flex;
  margin-bottom: 10px;
}

.info-label {
  font-weight: bold;
  margin-right: 10px;
  color: #606266;
  width: 150px; /* 统一标签宽度 */
  flex-shrink: 0;
}

.info-value {
  color: #303133;
  flex-grow: 1;
  word-break: break-all; /* 确保长文本能正常换行 */
}
</style>