# Meta风格QSS组件功能完整列表

## 基础容器组件

| 组件 | 功能描述 | 样式特点 | 使用场景 |
|------|----------|----------|----------|
| **QWidget** | 全局基础容器 | 浅灰背景 `#f0f2f5`，Meta标准背景色 | 应用主背景 |
| **QMainWindow** | 主窗口容器 | 与QWidget相同背景 | 应用主窗口 |
| **QDialog** | 对话框容器 | 与QWidget相同背景 | 弹窗、设置页面 |
| **QFrame** | 框架容器 | 白色背景，12px圆角，淡边框 | 卡片式内容区域 |

## 按钮系列组件

| 按钮类型 | CSS类名 | 样式特点 | 使用场景 |
|----------|---------|----------|----------|
| **主要按钮** | 默认 | Facebook蓝，白字，粗体，8px圆角 | 确认、提交、保存 |
| **次要按钮** | `class="secondary"` | 白色背景，灰边框，深色文字 | 取消、返回 |
| **幽灵按钮** | `class="ghost"` | 透明背景，蓝色文字，hover变半透明蓝 | 链接式操作 |
| **危险按钮** | `class="danger"` | 红色背景，白字 | 删除、清空等危险操作 |

### 按钮交互效果
- Hover：颜色变深 + 轻微上移
- Press：颜色更深 + 恢复位置
- Disabled：灰色背景，灰色文字

## 输入控件组件

| 组件 | 功能描述 | 样式特点 | 交互效果 |
|------|----------|----------|----------|
| **QLineEdit** | 单行文本输入 | 白色背景，8px圆角，灰边框 | 焦点时蓝色边框 |
| **QTextEdit** | 多行文本输入 | 同QLineEdit | 支持富文本编辑 |
| **QPlainTextEdit** | 纯文本多行输入 | 同QLineEdit | 纯文本模式 |

### 输入框特性
- 占位符文字：柔和灰色
- 选中文本：蓝色背景白字
- 禁用状态：浅灰背景

## 文本标签组件

| 标签类型 | CSS类名 | 字体大小 | 字重 | 使用场景 |
|----------|---------|----------|------|----------|
| **普通标签** | 默认 | 14px | 正常 | 一般文字说明 |
| **主标题** | `class="header"` | 24px | 700 | 页面主标题 |
| **副标题** | `class="subheader"` | 20px | 600 | 章节标题 |
| **正文** | `class="body"` | 15px | 正常 | 内容正文 |
| **说明文字** | `class="caption"` | 13px | 正常 | 辅助说明 |

## 选择控件组件

| 组件 | 功能描述 | 样式特点 | 交互效果 |
|------|----------|----------|----------|
| **QCheckBox** | 多选框 | 20×20px，6px圆角，勾选图标 | Hover时边框变蓝，背景微蓝 |
| **QRadioButton** | 单选框 | 20×20px圆形，选中时内圆填充 | 同CheckBox交互效果 |
| **QComboBox** | 下拉选择框 | 与输入框样式一致，下拉箭头 | 下拉面板8px圆角 |

## 数据展示组件

| 组件 | 功能描述 | 样式特点 | 特殊功能 |
|------|----------|----------|----------|
| **QListView** | 列表视图 | 白色卡片，12px圆角，项目8px圆角 | 选中项蓝色半透明背景 |
| **QListWidget** | 列表控件 | 同QListView | 更简单的API |
| **QTreeView** | 树形视图 | 同列表样式，展开/折叠箭头 | 层级显示 |
| **QTreeWidget** | 树形控件 | 同QTreeView | 更简单的API |

## 控制器组件

| 组件 | 功能描述 | 样式特点 | 交互特性 |
|------|----------|----------|----------|
| **QSlider** | 滑动条 | 6px轨道，20px圆形手柄，蓝色主题 | Hover时手柄放大 |
| **QSpinBox** | 数字输入框 | 输入框样式 + 上下箭头 | 箭头按钮Hover变灰 |
| **QDoubleSpinBox** | 浮点数输入框 | 同QSpinBox | 支持小数 |
| **QProgressBar** | 进度条 | 8px高度，圆角，蓝色进度 | 平滑动画效果 |

## 布局组件

| 组件 | 功能描述 | 样式特点 | 应用场景 |
|------|----------|----------|----------|
| **QTabWidget** | 标签页容器 | 白色面板，无边框 | 多页面切换 |
| **QTabBar** | 标签栏 | 透明背景，选中项蓝色半透明 | 标签页导航 |
| **QGroupBox** | 分组框 | 白色背景，12px圆角，标题浮动 | 功能分组 |
| **QSplitter** | 分割器 | 2px宽度，灰色，Hover变蓝 | 面板大小调整 |

## 导航组件

| 组件 | 功能描述 | 样式特点 | 交互效果 |
|------|----------|----------|----------|
| **QMenuBar** | 菜单栏 | 白色背景，底部边框 | 项目Hover变浅灰 |
| **QMenu** | 下拉菜单 | 白色背景，12px圆角，阴影 | 项目8px圆角，Hover变浅灰 |

## 滚动组件

| 组件 | 功能描述 | 样式特点 | 现代特性 |
|------|----------|----------|----------|
| **QScrollBar** | 滚动条 | 8px宽度，透明背景，圆角手柄 | 半透明设计，更现代 |

## 反馈组件

| 组件 | 功能描述 | 样式特点 | 显示效果 |
|------|----------|----------|----------|
| **QToolTip** | 工具提示 | 深色背景，白字，8px圆角 | 半透明浮动显示 |

## 使用示例代码

```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class MetaStyleDemo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meta风格QSS组件演示 - PySide6")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央控件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建标签页控件
        tab_widget = QTabWidget()
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(tab_widget)
        
        # 添加各个演示页面
        tab_widget.addTab(self.create_buttons_page(), "🔘 按钮组件")
        tab_widget.addTab(self.create_inputs_page(), "📝 输入组件") 
        tab_widget.addTab(self.create_selections_page(), "☑️ 选择组件")
        tab_widget.addTab(self.create_data_page(), "📊 数据组件")
        tab_widget.addTab(self.create_controls_page(), "🎛️ 控制组件")
        tab_widget.addTab(self.create_layout_page(), "📑 布局组件")
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.statusBar().showMessage("Meta风格QSS演示程序 - 所有组件展示")

    def create_menu_bar(self):
        """创建菜单栏演示"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("新建", self.on_action_clicked)
        file_menu.addAction("打开", self.on_action_clicked)
        file_menu.addAction("保存", self.on_action_clicked)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        edit_menu.addAction("复制", self.on_action_clicked)
        edit_menu.addAction("粘贴", self.on_action_clicked)
        edit_menu.addAction("查找", self.on_action_clicked)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        help_menu.addAction("关于", self.on_action_clicked)

    def create_buttons_page(self):
        """创建按钮演示页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 页面标题
        title = QLabel("按钮组件演示")
        title.setProperty("class", "header")
        layout.addWidget(title)
        
        # 按钮组
        button_group = QGroupBox("不同类型的按钮")
        button_layout = QGridLayout(button_group)
        
        # 主要按钮
        primary_btn = QPushButton("主要按钮 (Primary)")
        primary_btn.clicked.connect(lambda: self.show_message("点击了主要按钮"))
        button_layout.addWidget(primary_btn, 0, 0)
        
        # 次要按钮
        secondary_btn = QPushButton("次要按钮 (Secondary)")
        secondary_btn.setProperty("class", "secondary")
        secondary_btn.clicked.connect(lambda: self.show_message("点击了次要按钮"))
        button_layout.addWidget(secondary_btn, 0, 1)
        
        # 幽灵按钮
        ghost_btn = QPushButton("幽灵按钮 (Ghost)")
        ghost_btn.setProperty("class", "ghost")
        ghost_btn.clicked.connect(lambda: self.show_message("点击了幽灵按钮"))
        button_layout.addWidget(ghost_btn, 1, 0)
        
        # 危险按钮
        danger_btn = QPushButton("危险按钮 (Danger)")
        danger_btn.setProperty("class", "danger")
        danger_btn.clicked.connect(lambda: self.show_message("点击了危险按钮"))
        button_layout.addWidget(danger_btn, 1, 1)
        
        # 禁用按钮
        disabled_btn = QPushButton("禁用按钮 (Disabled)")
        disabled_btn.setEnabled(False)
        button_layout.addWidget(disabled_btn, 2, 0)
        
        # 带图标的按钮
        icon_btn = QPushButton("📁 图标按钮")
        button_layout.addWidget(icon_btn, 2, 1)
        
        layout.addWidget(button_group)
        
        # 文字标签演示
        label_group = QGroupBox("文字标签层次")
        label_layout = QVBoxLayout(label_group)
        
        header_label = QLabel("这是主标题 (Header)")
        header_label.setProperty("class", "header")
        label_layout.addWidget(header_label)
        
        subheader_label = QLabel("这是副标题 (Subheader)")
        subheader_label.setProperty("class", "subheader")
        label_layout.addWidget(subheader_label)
        
        body_label = QLabel("这是正文内容 (Body) - 用于显示主要内容文字，支持中英文混合显示效果。")
        body_label.setProperty("class", "body")
        body_label.setWordWrap(True)
        label_layout.addWidget(body_label)
        
        caption_label = QLabel("这是说明文字 (Caption) - 用于显示辅助信息")
        caption_label.setProperty("class", "caption")
        label_layout.addWidget(caption_label)
        
        normal_label = QLabel("这是普通标签 (Normal) - 默认样式")
        label_layout.addWidget(normal_label)
        
        layout.addWidget(label_group)
        layout.addStretch()
        
        return page

    def create_inputs_page(self):
        """创建输入组件演示页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("输入组件演示")
        title.setProperty("class", "header")
        layout.addWidget(title)
        
        # 输入框组
        input_group = QGroupBox("文本输入控件")
        input_layout = QFormLayout(input_group)
        
        # 单行输入框
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("请输入单行文本...")
        input_layout.addRow("单行输入框:", line_edit)
        
        # 密码输入框
        password_edit = QLineEdit()
        password_edit.setEchoMode(QLineEdit.Password)
        password_edit.setPlaceholderText("请输入密码...")
        input_layout.addRow("密码输入框:", password_edit)
        
        # 只读输入框
        readonly_edit = QLineEdit("这是只读文本")
        readonly_edit.setReadOnly(True)
        input_layout.addRow("只读输入框:", readonly_edit)
        
        # 禁用输入框
        disabled_edit = QLineEdit("禁用状态")
        disabled_edit.setEnabled(False)
        input_layout.addRow("禁用输入框:", disabled_edit)
        
        layout.addWidget(input_group)
        
        # 多行文本组
        text_group = QGroupBox("多行文本控件")
        text_layout = QHBoxLayout(text_group)
        
        # 富文本编辑器
        text_edit = QTextEdit()
        text_edit.setPlaceholderText("这是富文本编辑器，支持格式化文本...")
        text_edit.setMaximumHeight(120)
        text_layout.addWidget(text_edit)
        
        # 纯文本编辑器
        plain_text_edit = QPlainTextEdit()
        plain_text_edit.setPlaceholderText("这是纯文本编辑器，用于代码编辑...")
        plain_text_edit.setMaximumHeight(120)
        text_layout.addWidget(plain_text_edit)
        
        layout.addWidget(text_group)
        layout.addStretch()
        
        return page

    def create_selections_page(self):
        """创建选择组件演示页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("选择组件演示")
        title.setProperty("class", "header")
        layout.addWidget(title)
        
        # 复选框组
        checkbox_group = QGroupBox("复选框 (CheckBox)")
        checkbox_layout = QVBoxLayout(checkbox_group)
        
        checkbox1 = QCheckBox("选项1 - 苹果 🍎")
        checkbox1.setChecked(True)
        checkbox_layout.addWidget(checkbox1)
        
        checkbox2 = QCheckBox("选项2 - 香蕉 🍌")
        checkbox_layout.addWidget(checkbox2)
        
        checkbox3 = QCheckBox("选项3 - 橙子 🍊 (禁用)")
        checkbox3.setEnabled(False)
        checkbox_layout.addWidget(checkbox3)
        
        layout.addWidget(checkbox_group)
        
        # 单选框组
        radio_group = QGroupBox("单选框 (RadioButton)")
        radio_layout = QVBoxLayout(radio_group)
        
        radio1 = QRadioButton("小杯 ☕")
        radio1.setChecked(True)
        radio_layout.addWidget(radio1)
        
        radio2 = QRadioButton("中杯 ☕☕")
        radio_layout.addWidget(radio2)
        
        radio3 = QRadioButton("大杯 ☕☕☕")
        radio_layout.addWidget(radio3)
        
        radio4 = QRadioButton("超大杯 (禁用)")
        radio4.setEnabled(False)
        radio_layout.addWidget(radio4)
        
        layout.addWidget(radio_group)
        
        # 下拉选择框组
        combo_group = QGroupBox("下拉选择框 (ComboBox)")
        combo_layout = QFormLayout(combo_group)
        
        # 普通下拉框
        combo1 = QComboBox()
        combo1.addItems(["北京", "上海", "广州", "深圳", "杭州", "成都"])
        combo1.setCurrentText("上海")
        combo_layout.addRow("选择城市:", combo1)
        
        # 可编辑下拉框
        combo2 = QComboBox()
        combo2.setEditable(True)
        combo2.addItems(["Python", "Java", "JavaScript", "C++", "Go"])
        combo2.setCurrentText("Python")
        combo_layout.addRow("编程语言:", combo2)
        
        # 禁用下拉框
        combo3 = QComboBox()
        combo3.addItems(["选项1", "选项2", "选项3"])
        combo3.setEnabled(False)
        combo_layout.addRow("禁用状态:", combo3)
        
        layout.addWidget(combo_group)
        layout.addStretch()
        
        return page

    def create_data_page(self):
        """创建数据展示组件演示页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("数据展示组件")
        title.setProperty("class", "header")
        layout.addWidget(title)
        
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 列表视图
        list_group = QGroupBox("列表视图 (ListView)")
        list_layout = QVBoxLayout(list_group)
        
        list_widget = QListWidget()
        list_items = [
            "📁 文档文件夹",
            "📷 图片文件夹", 
            "🎵 音乐文件夹",
            "🎬 视频文件夹",
            "💾 下载文件夹",
            "🗑️ 回收站"
        ]
        for item in list_items:
            list_widget.addItem(item)
        list_widget.setCurrentRow(1)
        list_layout.addWidget(list_widget)
        
        splitter.addWidget(list_group)
        
        # 树形视图
        tree_group = QGroupBox("树形视图 (TreeView)")
        tree_layout = QVBoxLayout(tree_group)
        
        tree_widget = QTreeWidget()
        tree_widget.setHeaderLabels(["名称", "类型", "大小"])
        
        # 添加根节点
        root1 = QTreeWidgetItem(tree_widget, ["📁 项目文件夹", "文件夹", "--"])
        child1_1 = QTreeWidgetItem(root1, ["📄 main.py", "Python文件", "2.5KB"])
        child1_2 = QTreeWidgetItem(root1, ["📄 config.json", "JSON文件", "1.2KB"])
        child1_3 = QTreeWidgetItem(root1, ["📁 assets", "文件夹", "--"])
        child1_3_1 = QTreeWidgetItem(child1_3, ["🖼️ logo.png", "图片", "45KB"])
        child1_3_2 = QTreeWidgetItem(child1_3, ["🎵 sound.mp3", "音频", "3.2MB"])
        
        root2 = QTreeWidgetItem(tree_widget, ["📁 文档", "文件夹", "--"])
        child2_1 = QTreeWidgetItem(root2, ["📝 readme.md", "Markdown", "800B"])
        child2_2 = QTreeWidgetItem(root2, ["📊 report.xlsx", "Excel", "156KB"])
        
        tree_widget.expandAll()
        tree_layout.addWidget(tree_widget)
        
        splitter.addWidget(tree_group)
        layout.addWidget(splitter)
        
        return page

    def create_controls_page(self):
        """创建控制组件演示页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("控制组件演示")
        title.setProperty("class", "header")
        layout.addWidget(title)
        
        # 滑块组
        slider_group = QGroupBox("滑块控件 (Slider)")
        slider_layout = QFormLayout(slider_group)
        
        # 水平滑块
        h_slider = QSlider(Qt.Horizontal)
        h_slider.setRange(0, 100)
        h_slider.setValue(30)
        h_slider_label = QLabel("30")
        h_slider.valueChanged.connect(lambda v: h_slider_label.setText(str(v)))
        slider_layout.addRow("音量控制:", h_slider)
        slider_layout.addRow("当前值:", h_slider_label)
        
        layout.addWidget(slider_group)
        
        # 数字输入框组
        spin_group = QGroupBox("数字输入框 (SpinBox)")
        spin_layout = QFormLayout(spin_group)
        
        # 整数输入框
        spin_box = QSpinBox()
        spin_box.setRange(0, 999)
        spin_box.setValue(42)
        spin_box.setSuffix(" 个")
        spin_layout.addRow("数量:", spin_box)
        
        # 浮点数输入框
        double_spin = QDoubleSpinBox()
        double_spin.setRange(0.0, 100.0)
        double_spin.setValue(3.14)
        double_spin.setDecimals(2)
        double_spin.setSuffix(" cm")
        spin_layout.addRow("长度:", double_spin)
        
        layout.addWidget(spin_group)
        
        # 进度条组
        progress_group = QGroupBox("进度条 (ProgressBar)")
        progress_layout = QVBoxLayout(progress_group)
        
        # 普通进度条
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(65)
        progress_layout.addWidget(progress_bar)
        
        # 无限进度条
        infinite_progress = QProgressBar()
        infinite_progress.setRange(0, 0)  # 无限模式
        progress_layout.addWidget(infinite_progress)
        
        # 进度控制按钮
        progress_control_layout = QHBoxLayout()
        
        start_btn = QPushButton("开始")
        start_btn.clicked.connect(lambda: self.animate_progress(progress_bar))
        progress_control_layout.addWidget(start_btn)
        
        reset_btn = QPushButton("重置")
        reset_btn.setProperty("class", "secondary")
        reset_btn.clicked.connect(lambda: progress_bar.setValue(0))
        progress_control_layout.addWidget(reset_btn)
        
        progress_layout.addLayout(progress_control_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        return page

    def create_layout_page(self):
        """创建布局组件演示页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("布局组件演示")
        title.setProperty("class", "header")
        layout.addWidget(title)
        
        # 分组框演示
        group1 = QGroupBox("用户信息")
        group1_layout = QFormLayout(group1)
        
        name_edit = QLineEdit("张三")
        group1_layout.addRow("姓名:", name_edit)
        
        email_edit = QLineEdit("zhangsan@example.com")
        group1_layout.addRow("邮箱:", email_edit)
        
        age_spin = QSpinBox()
        age_spin.setRange(1, 120)
        age_spin.setValue(25)
        group1_layout.addRow("年龄:", age_spin)
        
        layout.addWidget(group1)
        
        # 嵌套标签页
        inner_tabs = QTabWidget()
        
        # 设置标签页1
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        theme_group = QGroupBox("主题设置")
        theme_layout = QVBoxLayout(theme_group)
        
        light_radio = QRadioButton("浅色主题 ☀️")
        light_radio.setChecked(True)
        theme_layout.addWidget(light_radio)
        
        dark_radio = QRadioButton("深色主题 🌙")
        theme_layout.addWidget(dark_radio)
        
        auto_radio = QRadioButton("跟随系统 🔄")
        theme_layout.addWidget(auto_radio)
        
        settings_layout.addWidget(theme_group)
        
        # 通知设置
        notify_group = QGroupBox("通知设置")
        notify_layout = QVBoxLayout(notify_group)
        
        email_notify = QCheckBox("邮件通知")
        email_notify.setChecked(True)
        notify_layout.addWidget(email_notify)
        
        push_notify = QCheckBox("推送通知")
        notify_layout.addWidget(push_notify)
        
        sound_notify = QCheckBox("声音提醒")
        sound_notify.setChecked(True)
        notify_layout.addWidget(sound_notify)
        
        settings_layout.addWidget(notify_group)
        settings_layout.addStretch()
        
        inner_tabs.addTab(settings_tab, "⚙️ 设置")
        
        # 关于标签页
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        app_info = QLabel("Meta风格QSS演示程序")
        app_info.setProperty("class", "subheader")
        about_layout.addWidget(app_info)
        
        version_info = QLabel("版本: 1.0.0")
        about_layout.addWidget(version_info)
        
        desc_info = QLabel("这是一个展示Meta风格QSS样式的演示程序，包含了PySide6所有常用组件的样式展示。")
        desc_info.setProperty("class", "body")
        desc_info.setWordWrap(True)
        about_layout.addWidget(desc_info)
        
        about_layout.addStretch()
        
        inner_tabs.addTab(about_tab, "ℹ️ 关于")
        
        layout.addWidget(inner_tabs)
        
        return page

    def show_message(self, message):
        """显示消息提示"""
        QMessageBox.information(self, "提示", message)

    def on_action_clicked(self):
        """菜单项点击事件"""
        action = self.sender()
        self.statusBar().showMessage(f"点击了菜单项: {action.text()}", 3000)

    def animate_progress(self, progress_bar):
        """动画演示进度条"""
        # 创建动画
        self.animation = QPropertyAnimation(progress_bar, b"value")
        self.animation.setDuration(3000)  # 3秒
        self.animation.setStartValue(0)
        self.animation.setEndValue(100)
        self.animation.start()

def main():
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("Meta风格QSS演示")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Demo")
    
    # 尝试加载QSS样式文件
    try:
        with open("style.qss", "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
            print("成功加载 style.qss")
    except FileNotFoundError:
        print("未找到 style.qss，界面将使用默认风格")
        print("💡 请确保 style.qss 文件在程序同目录下")
    except Exception as e:
        print(f"加载样式文件出错: {e}")
    
    # 创建并显示主窗口
    window = MetaStyleDemo()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())

```