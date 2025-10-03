<template>
  <div>
    <h1>Welcome to Tushare-DuckDB Dashboard!</h1>
    <p>This is the home page content.</p>
    <KlineChart /> <!-- 在这里添加 KlineChart 组件 -->
  </div>
</template>

<script setup lang="ts">
import KlineChart from '../components/KlineChart.vue'; // 导入 KlineChart 组件
</script>

<style scoped>
/* Styles for HomeView.vue */
</style>