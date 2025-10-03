<script setup lang="ts">
import HelloWorld from './components/HelloWorld.vue'
import { Location, Menu as IconMenu, Document, Setting } from '@element-plus/icons-vue'

const handleOpen = (key: string, keyPath: string[]) => {
  console.log(key, keyPath)
}
const handleClose = (key: string, keyPath: string[]) => {
  console.log(key, keyPath)
}
</script>

<template>
  <el-container class="common-layout">
    <el-header class="header">
      <div class="logo">Tushare-DuckDB</div>
      <div class="user-info">
        <!-- User related info or controls can go here -->
      </div>
    </el-header>
    <el-container>
      <el-aside width="200px" class="aside">
        <!-- Sidebar content, e.g., navigation menu -->
        <el-menu
          default-active="2"
          class="el-menu-vertical-demo"
          @open="handleOpen"
          @close="handleClose"
        >
          <el-sub-menu index="1">
            <template #title>
              <el-icon><location /></el-icon>
              <span>Navigator One</span>
            </template>
            <el-menu-item-group title="Group One">
              <el-menu-item index="1-1">item one</el-menu-item>
              <el-menu-item index="1-2">item two</el-menu-item>
            </el-menu-item-group>
            <el-menu-item-group title="Group Two">
              <el-menu-item index="1-3">item three</el-menu-item>
            </el-menu-item-group>
            <el-sub-menu index="1-4">
              <template #title>item four</template>
              <el-menu-item index="1-4-1">item one</el-menu-item>
            </el-sub-menu>
          </el-sub-menu>
          <el-menu-item index="2">
            <el-icon><icon-menu /></el-icon>
            <span>Navigator Two</span>
          </el-menu-item>
          <el-menu-item index="3" disabled>
            <el-icon><document /></el-icon>
            <span>Navigator Three</span>
          </el-menu-item>
          <el-menu-item index="4">
            <el-icon><setting /></el-icon>
            <span>Navigator Four</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main class="main-content">
        <!-- Main content area where router views will be displayed -->
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.common-layout {
  height: 100vh; /* Full viewport height */
}

.header {
  background-color: #409eff; /* Element Plus primary blue */
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
}

.logo {
  font-size: 24px;
  font-weight: bold;
}

.aside {
  background-color: #f4f4f5; /* Light gray for sidebar */
  padding-top: 20px;
}

.main-content {
  background-color: #ffffff; /* White for main content */
  padding: 20px;
  flex: 1; /* 新增：使其填充剩余垂直空间 */
}
</style>
