from python.app.ui import Matching_ui as UI
from PyQt5.QtWidgets import QApplication, QSplashScreen, QFileDialog, QMessageBox
import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from lsmatching import Matching
import os
import subprocess
import cv2

def show_info(parent, title="提示", message="操作成功"):
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.setIcon(QMessageBox.Information)
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()

class matching_app:
    working_directory = None
    left_img_path = None
    right_img_path = None
    left_points = None
    right_points = None
    h0 = None
    h1 = None
    a0 = None
    a1 = None
    a2 = None
    b0 = None
    b1 = None
    b2 = None
    
    @staticmethod
    def detect_gpu():
        try:
            subprocess.run(
                ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.ui = UI()
        if self.detect_gpu():
            self.ui.gpu_combo.addItem("GPU")
        self.ui.show()
        self.ui.directory_button.clicked.connect(self.set_directory)
        self.ui.left_btn.clicked.connect(self.get_left_image)
        self.ui.right_btn.clicked.connect(self.get_right_image)
        self.ui.choose_btn.clicked.connect(self.choose_cal)
        self.ui.matching_btn.clicked.connect(self.matching_cal)
        self.ui.output_btn.clicked.connect(self.matching_output)
        self.ui.param_btn.clicked.connect(self.check_params)

    def get_points(self):
        left_path = os.path.join(self.working_directory, "Left_result.dat")
        right_path = os.path.join(self.working_directory, "Right_result.dat")
        self.left_points = []
        self.right_points = []
        self.right_points_corrected = []
        with open(left_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    x, y = line.split('\t')
                    self.left_points.append((int(x), int(y)))
        with open(right_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    x, y = line.split('\t')
                    self.right_points.append((int(x), int(y)))

    def set_directory(self):
        folder = QFileDialog.getExistingDirectory(self.ui, "选择工作空间")
        if folder:
            self.working_directory = folder
            self.ui.directory_label.setText(folder)

    def get_left_image(self):
        if not self.working_directory:
            show_info(self.ui, title = "提示", message= "请先选择工作空间！")
            return
        self.left_img_path, _ = QFileDialog.getOpenFileName(
                                                            self.ui,
                                                            "选择左影像",
                                                            self.working_directory,
                                                            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
                                                        )
        if not self.left_img_path:
            return
        self.ui.left_img.set_image(self.left_img_path)

    def get_right_image(self):
        if not self.working_directory:
            show_info(self.ui, title = "提示", message= "请先选择工作空间！")
            return
        self.right_img_path, _ = QFileDialog.getOpenFileName(
                                                            self.ui,
                                                            "选择右影像",
                                                            self.working_directory,
                                                            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
                                                        )
        if not self.right_img_path:
            return
        self.ui.right_img.set_image(self.right_img_path)

    def choose_cal(self):
        if self.ui.left_img.click_point is None or self.ui.right_img.click_point is None:
            show_info(self.ui, message="请先选择同名点！")
            return
        matching = Matching(self.left_img_path, self.right_img_path)
        x1, y1 = self.ui.left_img.click_point
        x2, y2 = self.ui.right_img.click_point
        windowsize_text = self.ui.windowsize_line_edit.text()
        ncc_thres_text = self.ui.d_ncc_line_edit.text()
        windowsize = int(windowsize_text) if windowsize_text else 15
        ncc_value = float(ncc_thres_text) if ncc_thres_text else 0.04
        matching.set_params(windowsize, ncc_value)
        matching.set_centers(x1, y1, x2, y2)
        left_window = matching.get_left_window()
        right_window = matching.get_right_window()
        matching.calculate()
        left_window_matched = matching.get_left_window()
        right_window_matched = matching.get_right_window()

        self.ui.left_window_origin.set_image(left_window)
        self.ui.right_window_origin.set_image(right_window)
        self.ui.left_window.set_image(left_window_matched)
        self.ui.right_window.set_image(right_window_matched)

        self.ui.me.setText(f"{matching.get_delta0():.6f}")
        self.ui.SNR.setText(f"{matching.get_SNR():.6f}")
        self.ui.rho.setText(f"{matching.get_rho():.6f}")
        self.ui.texture.setText(f"{matching.get_deltag_()/matching.get_deltag():.6f}")
        self.ui.deltax.setText(f"{matching.get_deltax():.6f}")

        self.ui.x1.setText(f"x={int(self.ui.left_img.click_point[0])}")
        self.ui.y1.setText(f"y={int(self.ui.left_img.click_point[1])}")
        self.ui.x2.setText(f"x={int(self.ui.right_img.click_point[0])}")
        self.ui.y2.setText(f"y={int(self.ui.right_img.click_point[1])}")
        self.ui.x3.setText(f"x={int(self.ui.left_img.click_point[0])}")
        self.ui.y3.setText(f"y={int(self.ui.left_img.click_point[1])}")
        self.ui.x4.setText(f"x={matching.get_matched_x():.6f}")
        self.ui.y4.setText(f"y={matching.get_matched_y():.6f}")
        
        self.h0 = matching.get_h0()
        self.h1 = matching.get_h1()
        self.a0 = matching.get_a0()
        self.a1 = matching.get_a1()
        self.a2 = matching.get_a2()
        self.b0 = matching.get_b0()
        self.b1 = matching.get_b1()
        self.b2 = matching.get_b2()

    def check_params(self):
        show_info(self.ui, "查看参数", f"h0 = {self.h0:.6f}, h1 = {self.h1:.6f}, a0 = {self.a0:.6f}, a1 = {self.a1:.6f}, a2 = {self.a2:.6f}, b0 = {self.b0:.6f}, b1 = {self.b1:.6f}, b2 = {self.b2:.6f}")

    def matching_cal(self):
        self.right_points_corrected = []
        
        use_gpu = self.ui.gpu_combo.currentText() == "GPU"
        matching = Matching(self.left_img_path, self.right_img_path)
        windowsize_text = self.ui.windowsize_line_edit.text()
        ncc_thres_text = self.ui.d_ncc_line_edit.text()
        windowsize = int(windowsize_text) if windowsize_text else 35
        ncc_value = float(ncc_thres_text) if ncc_thres_text else 0.04
        
        matching_windowsize_text = self.ui.windowsize_line_edit2.text()
        matching_ncc_thres_text = self.ui.ncc_line_edit2.text()
        matching_windowsize = int(matching_windowsize_text) if matching_windowsize_text else 5
        matching_ncc_value = float(matching_ncc_thres_text) if matching_ncc_thres_text else 0.3
            
        if use_gpu:
            raw = matching.batch_adjust_gpu(window_size=windowsize,
                                            d_corr=ncc_value,
                                            max_iter=20,
                                            matching_wsize = matching_windowsize,
                                            corr_threshold=matching_ncc_value,
                                            savepath=self.working_directory
                                            )
            self.right_points_corrected = [
                (rx, ry) for _, _, rx, ry in raw
                if rx > 0 and ry > 0
            ]
            
            x_path = os.path.join(self.working_directory, "Left_result_corrected.dat")
            left_img = cv2.imread(self.left_img_path)
            with open(x_path, 'w') as file:
                for (lx, ly, _, _) in raw:
                    file.write(f"{lx}\t{ly}\n")
                    cv2.circle(left_img, (lx, ly), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.working_directory, "left_marked.jpg"), left_img)
        else:
            matching.set_params(windowsize, ncc_value)
            windowsize_text = None
            ncc_thres_text = None

            matching.set_matching_params(matching_windowsize, matching_ncc_value)
            matching.get_matched_points(self.working_directory)
            self.get_points()
            for index, _ in enumerate(self.left_points):
                x1, y1 = self.left_points[index]
                x2, y2 = self.right_points[index]
                matching.set_centers(x1, y1, x2, y2)
                matching.calculate()
                self.right_points_corrected.append((matching.get_matched_x(), matching.get_matched_y()))

    def matching_output(self):
        corrected_y_path = os.path.join(self.working_directory, "Right_result_corrected.dat")
        right_img = cv2.imread(self.right_img_path)
        with open(corrected_y_path, 'w') as file:
            for (x, y) in self.right_points_corrected:
                file.write(f"{x}\t{y}\n")
                cv2.circle(right_img, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.working_directory, "right_marked.jpg"), right_img)
    
    def run(self):
        sys.exit(self.app.exec_())