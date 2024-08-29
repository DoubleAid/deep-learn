# ROS

## 1. 安装 ROS2

### 1.1 设置 ubuntu 源

```shell
apt update && sudo apt upgrade -y
apt install curl gnupg2 lsb-release

# 添加 ROS2 的 apt 软件源
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sh -c 'echo "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
```

### 1.2 安装ROS

```bash
sudo apt update
sudo apt install ros-humble-desktop
```

### 1.3 配置环境

```bash
source /opt/ros/humble/setup.bash
```
