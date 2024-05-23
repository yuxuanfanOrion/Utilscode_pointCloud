import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QLabel, QGridLayout, QDoubleSpinBox, QGroupBox
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

class PointCloudRegistrationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.source_cloud = None
        self.target_cloud = None
        self.transformation = None

        self.initUI()
        self.generate_random_point_clouds()
        self.display_random_transformation()

    def initUI(self):
        self.setWindowTitle('点云配准算法可视化界面')
        self.setGeometry(100, 100, 1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧布局：点云显示和加载
        left_layout = QVBoxLayout()

        # 标题
        title_label = QLabel('点云配准算法可视化界面')
        title_label.setFont(QFont('Arial', 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_label)

        # 点云显示
        point_cloud_display_group = QGroupBox('点云显示')
        point_cloud_display_layout = QHBoxLayout()
        self.target_cloud_display = QLabel()
        self.source_cloud_display = QLabel()
        self.target_cloud_display.setFixedSize(400, 400)
        self.source_cloud_display.setFixedSize(400, 400)
        point_cloud_display_layout.addWidget(self.target_cloud_display)
        point_cloud_display_layout.addWidget(self.source_cloud_display)
        point_cloud_display_group.setLayout(point_cloud_display_layout)
        left_layout.addWidget(point_cloud_display_group)

        # 加载按钮
        load_buttons_group = QGroupBox('加载点云')
        load_buttons_layout = QHBoxLayout()
        self.load_target_button = QPushButton('导入目标点云')
        self.load_source_button = QPushButton('导入源点云')
        self.load_target_button.clicked.connect(self.load_target_cloud)
        self.load_source_button.clicked.connect(self.load_source_cloud)
        load_buttons_layout.addWidget(self.load_target_button)
        load_buttons_layout.addWidget(self.load_source_button)
        load_buttons_group.setLayout(load_buttons_layout)
        left_layout.addWidget(load_buttons_group)

        # 处理和配准选项
        options_group = QGroupBox('处理和配准选项')
        options_layout = QGridLayout()

        self.target_voxel_size_spinbox = QDoubleSpinBox()
        self.target_voxel_size_spinbox.setValue(2.0)
        self.target_max_z_spinbox = QDoubleSpinBox()
        self.target_max_z_spinbox.setValue(450)
        self.target_min_z_spinbox = QDoubleSpinBox()
        self.target_min_z_spinbox.setValue(390)

        options_layout.addWidget(QLabel('体素化网格大小:'), 0, 0)
        options_layout.addWidget(self.target_voxel_size_spinbox, 0, 1)
        options_layout.addWidget(QLabel('Zmax:'), 1, 0)
        options_layout.addWidget(self.target_max_z_spinbox, 1, 1)
        options_layout.addWidget(QLabel('Zmin:'), 2, 0)
        options_layout.addWidget(self.target_min_z_spinbox, 2, 1)

        self.manual_registration_button = QPushButton('手动配准')
        self.automatic_registration_button = QPushButton('自动配准')
        self.manual_registration_button.clicked.connect(self.manual_registration)
        self.automatic_registration_button.clicked.connect(self.automatic_registration)
        options_layout.addWidget(self.manual_registration_button, 3, 0, 1, 2)
        options_layout.addWidget(self.automatic_registration_button, 4, 0, 1, 2)

        self.visualize_registration_button = QPushButton('可视化配准')
        self.visualize_registration_button.clicked.connect(self.visualize_registration)
        options_layout.addWidget(self.visualize_registration_button, 5, 0, 1, 2)

        options_group.setLayout(options_layout)
        left_layout.addWidget(options_group)

        main_layout.addLayout(left_layout)

        # 右侧布局：配准结果
        right_layout = QVBoxLayout()

        results_group = QGroupBox('配准结果')
        results_layout = QVBoxLayout()
        self.result_label = QLabel('配准结果:')
        self.result_label.setFont(QFont('Arial', 12))
        self.transformation_matrix_label = QLabel('变换矩阵:')
        self.transformation_matrix_label.setFont(QFont('Arial', 12))
        self.rotation_matrix_label = QLabel('旋转矩阵 R:')
        self.rotation_matrix_label.setFont(QFont('Arial', 12))
        self.translation_vector_label = QLabel('平移向量 t:')
        self.translation_vector_label.setFont(QFont('Arial', 12))
        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.transformation_matrix_label)
        results_layout.addWidget(self.rotation_matrix_label)
        results_layout.addWidget(self.translation_vector_label)
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)

        main_layout.addLayout(right_layout)

    def generate_random_point_clouds(self):
        self.target_cloud = self.create_random_point_cloud()
        self.source_cloud = self.create_random_point_cloud()
        self.update_cloud_display(self.target_cloud, self.target_cloud_display)
        self.update_cloud_display(self.source_cloud, self.source_cloud_display)

    def create_random_point_cloud(self, num_points=1000):
        points = np.random.rand(num_points, 3)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.paint_uniform_color([1, 0, 0])  # 设置点云颜色为红色
        return cloud

    def load_target_cloud(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择目标点云文件", "", "Point Cloud Files (*.ply *.pcd *.xyz *.bin)", options=options)
        if file_name:
            self.target_cloud = self.read_point_cloud(file_name)
            self.update_cloud_display(self.target_cloud, self.target_cloud_display)
            print("目标点云加载成功!")

    def load_source_cloud(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择源点云文件", "", "Point Cloud Files (*.ply *.pcd *.xyz *.bin)", options=options)
        if file_name:
            self.source_cloud = self.read_point_cloud(file_name)
            self.update_cloud_display(self.source_cloud, self.source_cloud_display)
            print("源点云加载成功!")

    def read_point_cloud(self, file_name):
        if file_name.endswith('.bin'):
            return self.read_bin_point_cloud(file_name)
        else:
            cloud = o3d.io.read_point_cloud(file_name)
            cloud.paint_uniform_color([1, 0, 0])  # 设置点云颜色为红色
            return cloud

    def read_bin_point_cloud(self, file_name):
        points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        cloud.paint_uniform_color([1, 0, 0])  # 设置点云颜色为红色
        return cloud

    def update_cloud_display(self, cloud, display_label):
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(True)
        vis.destroy_window()

        plt.imshow(np.asarray(image))
        plt.axis('off')
        plt.savefig("temp.png")
        plt.close()

        pixmap = QPixmap("temp.png")
        display_label.setPixmap(pixmap.scaled(display_label.size(), aspectRatioMode=Qt.KeepAspectRatio))

    def display_random_transformation(self):
        random_transformation = self.create_random_transformation()
        self.transformation_matrix_label.setText(f'随机变换矩阵:\n{random_transformation}')
        self.rotation_matrix_label.setText(f'旋转矩阵 R:\n{random_transformation[:3, :3]}')
        self.translation_vector_label.setText(f'平移向量 t:\n{random_transformation[:3, 3]}')

    def create_random_transformation(self):
        random_transformation = np.eye(4)
        random_transformation[:3, :3] = np.random.rand(3, 3)
        random_transformation[:3, 3] = np.random.rand(3)
        return random_transformation

    def manual_registration(self):
        print("手动配准功能待实现")

    def automatic_registration(self):
        if self.source_cloud and self.target_cloud:
            threshold = 0.02
            trans_init = np.identity(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                self.source_cloud, self.target_cloud, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            self.transformation = reg_p2p.transformation
            self.result_label.setText(f'配准结果: \n配准精度: {reg_p2p.inlier_rmse:.4f}')
            self.transformation_matrix_label.setText(f'变换矩阵:\n{self.transformation}')
            self.rotation_matrix_label.setText(f'旋转矩阵 R:\n{self.transformation[:3, :3]}')
            self.translation_vector_label.setText(f'平移向量 t:\n{self.transformation[:3, 3]}')
            print("点云自动配准完成!")

    def visualize_registration(self):
        if self.source_cloud and self.target_cloud and self.transformation is not None:
            source_temp = self.source_cloud.transform(self.transformation)
            o3d.visualization.draw_geometries([source_temp, self.target_cloud])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PointCloudRegistrationApp()
    ex.show()
    sys.exit(app.exec_())

