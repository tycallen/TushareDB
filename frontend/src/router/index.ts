import { createRouter, createWebHistory } from 'vue-router';
import HomeView from '../views/HomeView.vue';
import StockDetailView from '../views/StockDetailView.vue'; // 导入新的视图组件

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/stock/:ts_code', // 添加股票详情页路由
      name: 'stock-detail',
      component: StockDetailView
    }
  ]
});

export default router;