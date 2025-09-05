#!/bin/bash

# RDMA Core Debian 打包脚本 - 最简版
# 只包含核心逻辑

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


# 获取项目根目录
PROJECT_ROOT=$(pwd)
DEB_OUTPUT_DIR="$PROJECT_ROOT/deb"

log_info "开始RDMA Core Debian打包过程..."
log_info "项目目录: $PROJECT_ROOT"
log_info "输出目录: $DEB_OUTPUT_DIR"

# 创建输出目录
mkdir -p "$DEB_OUTPUT_DIR"

# 步骤1: 设置源码格式为native
log_info "步骤1: 设置源码格式为native..."
echo "3.0 (native)" > debian/source/format
log_success "已设置源码格式为native"

# 步骤2: 禁用Python绑定
log_info "步骤2: 禁用Python绑定..."
# 备份原始rules文件
cp debian/rules debian/rules.backup
# 修改rules文件，添加-DNO_PYVERBS=1选项
sed -i 's/-DPYTHON_EXECUTABLE:PATH=\/usr\/bin\/python3/-DNO_PYVERBS=1/' debian/rules
log_success "已禁用Python绑定"

# 步骤3: 执行打包
log_info "步骤3: 开始Debian打包..."
log_info "执行: dpkg-buildpackage -us -uc"
dpkg-buildpackage -us -uc
log_success "Debian打包完成！"

# 步骤4: 移动生成的deb包到指定目录
log_info "步骤4: 整理生成的deb包..."
log_info "注意: dpkg-buildpackage会在上级目录生成包文件，现在移动到 $DEB_OUTPUT_DIR"

# 移动主要功能包到输出目录（只保留.deb包，排除.ddeb调试包）
mv ../*.deb "$DEB_OUTPUT_DIR/" 2>/dev/null || true

# 清理调试符号包(.ddeb)和Python绑定包
rm -f ../*.ddeb
rm -f ../python3-pyverbs*.deb

# 清理其他生成的文件
rm -f ../rdma-core_*.build
rm -f ../rdma-core_*.buildinfo
rm -f ../rdma-core_*.changes
rm -f ../rdma-core_*.dsc
rm -f ../rdma-core_*.tar.*

# 显示生成的包
log_success "生成的deb包已保存到: $DEB_OUTPUT_DIR"
echo ""
log_info "生成的包列表:"
ls -la "$DEB_OUTPUT_DIR"/*.deb 2>/dev/null | while read line; do
    echo "  $line"
done

# 步骤5: 恢复原始配置
log_info "步骤5: 恢复原始配置..."
if [ -f "debian/rules.backup" ]; then
    mv debian/rules.backup debian/rules
    log_success "已恢复原始 debian/rules 文件"
fi

echo ""
log_success "打包过程完成！"
log_info "所有deb包已保存到: $DEB_OUTPUT_DIR"
log_info "你可以使用以下命令安装这些包:"
echo "  sudo dpkg -i $DEB_OUTPUT_DIR/*.deb"