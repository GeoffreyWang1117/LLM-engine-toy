# 部署指南

## GitHub Pages 部署

本项目已配置自动部署到GitHub Pages。当您推送代码到main或master分支时，GitHub Actions会自动构建WASM并部署。

### 步骤

1. **推送代码到GitHub**

```bash
git add .
git commit -m "Initial commit: LLM Engine Toy with WASM support"
git push origin main
```

2. **启用GitHub Pages**

- 进入GitHub仓库页面
- 点击 Settings
- 在左侧菜单找到 Pages
- Source选择 "GitHub Actions"
- 保存设置

3. **等待部署完成**

- 进入 Actions 标签页
- 查看工作流运行状态
- 等待构建和部署完成（约2-3分钟）

4. **访问网站**

部署完成后，您的网站将可以通过以下地址访问：

```
https://geoffreywang1117.github.io/LLM-engine-toy/
```

## 本地测试WASM版本

如果您想在本地测试WASM版本：

### 1. 安装Emscripten

```bash
# 克隆emsdk
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# 安装和激活最新版本
./emsdk install latest
./emsdk activate latest

# 设置环境变量（每次新终端都需要）
source ./emsdk_env.sh
```

### 2. 构建WASM

返回项目目录：

```bash
cd /path/to/LLM-engine-toy
./build_wasm.sh
```

### 3. 启动本地服务器

```bash
cd web
python3 -m http.server 8000
```

### 4. 在浏览器中访问

打开浏览器访问：
```
http://localhost:8000
```

## 常见问题

### Q: GitHub Actions构建失败？

A: 检查以下几点：
- 确保仓库有正确的权限设置
- 检查 .github/workflows/deploy.yml 文件是否正确
- 查看Actions日志了解具体错误信息

### Q: 网页显示"加载失败"？

A: 可能的原因：
- WASM文件路径不正确
- 浏览器不支持WebAssembly
- 需要通过HTTP服务器访问（不能直接打开HTML文件）

### Q: 如何更新网站？

A: 只需推送新代码到GitHub：
```bash
git add .
git commit -m "Update features"
git push origin main
```

GitHub Actions会自动重新构建和部署。

## 自定义域名（可选）

如果您想使用自定义域名：

1. 在仓库根目录创建 `CNAME` 文件，内容为您的域名：
```
www.yourdomain.com
```

2. 在域名提供商处设置DNS记录：
- 类型: CNAME
- 名称: www
- 值: geoffreywang1117.github.io

3. 推送更改到GitHub

4. 在GitHub仓库Settings > Pages中验证自定义域名
