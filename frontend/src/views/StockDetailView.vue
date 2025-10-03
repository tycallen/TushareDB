<template>
  <div class="stock-detail-page">
    <h1>股票详情页</h1>
    <KlineChart />
  </div>
</template>

<script setup lang="ts">
import KlineChart from '../components/KlineChart.vue';
</script>

<style scoped>
.stock-detail-page {
  width: 100%; /* 确保占据全部宽度 */
  height: 100%; /* 确保占据全部高度 */
  /* padding: 20px; */ /* 暂时移除 padding */
}
</style>