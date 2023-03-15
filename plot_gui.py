import sys
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QGroupBox, QGridLayout, QLabel, QColorDialog, QSplitter, QComboBox, QMenuBar, QMenu, QSizePolicy, QProgressBar, QWidget, QMainWindow, QScrollArea, QSlider
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation, FuncAnimation
from scipy.integrate import odeint

import numpy as np

stylesheet = '''

#TimeProgressBar {
}

QWidget {
    font-family: Ubuntu;
    font-size: 18px;
}

'''

class MainWindow(QMainWindow):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
            

        self.num = 0
        self.paused = False
        self.cog_shown = True
        self.anim_start = False
        self.trace_shown = True
        self.origin_shown = True
        self.axes_shown = False
        self.axes_ticks_shown = False
        self.grid_shown = False
        self.grid_labels_shown = False
        self.tb_col1 = "#FF5000"
        self.tb_col2 = "#563843"
        self.tb_cog_col = "#342482"

        self.time = 0

        self.anim_speed = 10
        
        self.setStyleSheet(stylesheet)
        
        self.fig = Figure(figsize=(10, 10), dpi=100)

        
        self.canvas = FigureCanvas(self.fig)
        self._main = QWidget()
        self.setCentralWidget(self._main)
        self.layout = QHBoxLayout(self._main)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_aspect('equal', 'box')
    

        # self.ax.w_xaxis.set_pane_color((0.75, 0.5, 0.3, 1))
        # self.ax.w_yaxis.set_pane_color((R,G,B,A))
        # self.ax.w_zaxis.set_pane_color((R,G,B,A))


        self.show()

        self.guiInit()
        self.tb_col1_preview.setStyleSheet("background: {}".format(self.tb_col1))
        self.tb_col2_preview.setStyleSheet("background: {}".format(self.tb_col2))
        self.tb_cog_col_preview.setStyleSheet("background: {}".format(self.tb_cog_col))
        self.plot_face_color_preview.setStyleSheet("background: {}".format("#FFFFFF"))

        
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_axis_off()
        self.ax.grid(False)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.animate_func)
        self.init_vals()
        self.calc()

    def toggle_cog(self):
        self.cog_shown = not self.cog_shown

    def toggle_trace(self):
        self.trace_shown = not self.trace_shown

    def toggle_origin(self):
        self.origin_shown = not self.origin_shown
            
    def toggle_axes(self):
        self.axes_shown = not self.axes_shown
        if self.axes_shown:
            self.axes_tick_box.setEnabled(True)
            self.grid_labels_box.setEnabled(True)
            self.grid_box.setEnabled(True)
        else:
            self.axes_tick_box.setEnabled(False)
            self.grid_labels_box.setEnabled(False)
            self.grid_box.setEnabled(False)


    def toggle_axes_ticks(self):
        self.axes_ticks_shown = not self.axes_ticks_shown
        if self.axes_ticks_shown:
            self.grid_box.setEnabled(True)
            self.grid_labels_box.setEnabled(True)
        else:
            self.grid_box.setEnabled(False)
            self.grid_labels_box.setEnabled(False)

    def toggle_grid(self):
        self.grid_shown = not self.grid_shown
        if self.grid_shown:
            self.grid_labels_box.setEnabled(True)
        else:
            self.grid_labels_box.setEnabled(False)

    def toggle_grid_labels(self):
        self.grid_labels_shown = not self.grid_labels_shown

    def anim_toggle(self):
        self.paused = not self.paused
        if self.paused:
            self.anim_toggle_button.setText("Play Animation")
        else:
            self.anim_toggle_button.setText("Pause Animation")
    
    def animate_func(self):
        if not self.paused:
            if self.num + self.anim_speed < self.T - 1:
                self.num += self.anim_speed
            else:
                self.num = 0
            
            self.timeProgressbar.setValue(abs(self.num))
            self.timeValue.setText("{} s".format(str(self.t[self.num])))
            self.ax.cla()

            if self.cog_shown:
                self.ax.scatter(self.cog_sol[self.num, 0], self.cog_sol[self.num, 1], self.cog_sol[self.num, 2], c=self.tb_cog_col, marker='o', s=100)

            if self.trace_shown:
                self.ax.plot3D(self.r1_sol[: self.num + 1, 0], self.r1_sol[: self.num + 1, 1], self.r1_sol[: self.num + 1, 2], c = self.tb_col1)
                self.ax.plot3D(self.r2_sol[: self.num + 1, 0], self.r2_sol[: self.num + 1, 1], self.r2_sol[: self.num + 1, 2], c = self.tb_col2)

            self.ax.scatter(self.r1_sol[self.num, 0], self.r1_sol[self.num, 1], self.r1_sol[self.num, 2], c= self.tb_col1, marker='o', s=self.radius1 * 100)

            self.ax.scatter(self.r2_sol[self.num, 0], self.r2_sol[self.num, 1], self.r2_sol[self.num, 2], c=self.tb_col2, marker='o', s=self.radius2 * 100)

            if self.origin_shown:
                self.ax.scatter(self.r1_sol[0, 0], self.r1_sol[0, 1], self.r1_sol[0, 2], c='black', marker='o', s=50)
                self.ax.scatter(self.r2_sol[0, 0], self.r2_sol[0, 1], self.r2_sol[0, 2], c='black', marker='o', s=50)

            if not self.grid_shown:
                self.ax.grid(False)

            if not self.axes_shown:
                self.ax.set_axis_off()

            if not self.grid_labels_shown:
                self.ax.set_xticklabels([])
                self.ax.set_yticklabels([])
                self.ax.set_zticklabels([])
            
            if not self.axes_ticks_shown:
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_zticks([])

            self.canvas.draw()

    # Function for starting and stopping animation
    def anim_start_stop(self):
        self.anim_start_stop_button.setText("Reset Animation")
        self.timer.start()
        self.get_inputs()
        self.calc()
        self.num = 0
        self.ax.cla()
    
    # Function for getting the inputs from lineedits
    def get_inputs(self):
        self.m1 = float(self.tb_m1.text())
        self.r1 = np.array([float(self.tb_r1x.text()), float(self.tb_r1y.text()), float(self.tb_r1z.text())])
        self.v1 = np.array([float(self.tb_v1x.text()), float(self.tb_v1y.text()), float(self.tb_v1z.text())])
        self.radius1 = 5

        self.m2 = float(self.tb_m2.text())
        self.r2 = np.array([float(self.tb_r2x.text()), float(self.tb_r2y.text()), float(self.tb_r2z.text())])
        self.v2 = np.array([float(self.tb_v2x.text()), float(self.tb_v2y.text()), float(self.tb_v2z.text())])
        self.radius2 = 2

        self.t = np.arange(float(self.time0.text()), float(self.timef.text()), float(self.timedt.text()))


    # Function for GUI Components
    def guiInit(self):
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()

        self.timeProgressbar = QProgressBar(objectName="TimeProgressBar")

        # Group Box

        self.anim_groupbox = QGroupBox("Animation")
        self.anim_groupbox_layout = QVBoxLayout(self.anim_groupbox)

        self.param_groupbox = QGroupBox("Parameters")
        self.param_groupbox_layout = QGridLayout(self.param_groupbox)

    
        # Parameter GroupBox Elements


        self.tb_m1 = QLineEdit("1e26")
        self.tb_v1_layout = QHBoxLayout()
        self.tb_r1_layout = QHBoxLayout()

        self.tb_m2 = QLineEdit("1e20")
        self.tb_v2_layout = QHBoxLayout()
        self.tb_r2_layout = QHBoxLayout()

        self.tb_v1x = QLineEdit("10")
        self.tb_v1y = QLineEdit("20")
        self.tb_v1z = QLineEdit("30")

        self.tb_v1_layout.addWidget(self.tb_v1x)
        self.tb_v1_layout.addWidget(self.tb_v1y)
        self.tb_v1_layout.addWidget(self.tb_v1z)

        self.tb_v2x = QLineEdit("0")
        self.tb_v2y = QLineEdit("40")
        self.tb_v2z = QLineEdit("0")

        self.tb_v2_layout.addWidget(self.tb_v2x)
        self.tb_v2_layout.addWidget(self.tb_v2y)
        self.tb_v2_layout.addWidget(self.tb_v2z)

        self.tb_r1x = QLineEdit("0")
        self.tb_r1y = QLineEdit("0")
        self.tb_r1z = QLineEdit("0")

        self.tb_r1_layout.addWidget(self.tb_r1x)
        self.tb_r1_layout.addWidget(self.tb_r1y)
        self.tb_r1_layout.addWidget(self.tb_r1z)

        self.tb_r2x = QLineEdit("0")
        self.tb_r2y = QLineEdit("3000")
        self.tb_r2z = QLineEdit("0")

        self.tb_r2_layout.addWidget(self.tb_r2x)
        self.tb_r2_layout.addWidget(self.tb_r2y)
        self.tb_r2_layout.addWidget(self.tb_r2z)

        self.tb_col1_button = QPushButton("Color 1")
        self.tb_col1_button.clicked.connect(self.get_col1)
        self.tb_col1_preview = QLabel(" ")

        self.tb_col2_button = QPushButton("Color 2")
        self.tb_col2_button.clicked.connect(self.get_col2)
        self.tb_col2_preview = QLabel(" ")

        self.tb_cog_col_button = QPushButton("COG Color")
        self.tb_cog_col_button.clicked.connect(self.get_cog_col)
        self.tb_cog_col_preview = QLabel(" ")
        
        self.time0 = QLineEdit("0")
        self.timef = QLineEdit("480")
        self.timedt = QLineEdit("0.5")

        self.time_layout = QHBoxLayout()
    
        self.time_layout.addWidget(self.time0)
        self.time_layout.addWidget(self.timef)
        self.time_layout.addWidget(self.timedt)

        self.param_groupbox_layout.addWidget(QLabel("Mass 1 "), 0, 0)
        self.param_groupbox_layout.addWidget(self.tb_m1, 0, 1)

        self.param_groupbox_layout.addWidget(QLabel("Position 1"), 1, 0)
        self.param_groupbox_layout.addLayout(self.tb_r1_layout, 1, 1)

        self.param_groupbox_layout.addWidget(QLabel("Velocity 1"), 2, 0)
        self.param_groupbox_layout.addLayout(self.tb_v1_layout, 2, 1)

        self.param_groupbox_layout.addWidget(self.tb_col1_button, 3, 0)
        self.param_groupbox_layout.addWidget(self.tb_col1_preview, 3, 1)

        self.param_groupbox_layout.addWidget(QLabel("Mass 2 "), 4, 0)
        self.param_groupbox_layout.addWidget(self.tb_m2, 4, 1)

        self.param_groupbox_layout.addWidget(QLabel("Position 1"), 5, 0)
        self.param_groupbox_layout.addLayout(self.tb_r2_layout, 5, 1)

        self.param_groupbox_layout.addWidget(QLabel("Velocity 2"), 6, 0)
        self.param_groupbox_layout.addLayout(self.tb_v2_layout, 6, 1)

        self.param_groupbox_layout.addWidget(self.tb_col2_button, 7, 0)
        self.param_groupbox_layout.addWidget(self.tb_col2_preview, 7, 1)

        self.param_groupbox_layout.addWidget(self.tb_cog_col_button, 8, 0)
        self.param_groupbox_layout.addWidget(self.tb_cog_col_preview, 8, 1)

        self.param_groupbox_layout.addWidget(QLabel("Time"), 9, 0)
        self.param_groupbox_layout.addLayout(self.time_layout, 9, 1)
        # Buttons

        self.anim_start_stop_button = QPushButton("Start Animation")
        self.anim_start_stop_button.clicked.connect(self.anim_start_stop)

        self.anim_toggle_button = QPushButton("Pause Animation")
        self.anim_toggle_button.clicked.connect(self.anim_toggle)

        self.cog_box = QCheckBox("Show Center of Mass")
        self.cog_box.setChecked(True)
        self.cog_box.clicked.connect(self.toggle_cog)

        self.trace_box = QCheckBox("Show trace")
        self.trace_box.setChecked(True)
        self.trace_box.clicked.connect(self.toggle_trace)

        self.origin_box = QCheckBox("Show Origin Points")
        self.origin_box.setChecked(True)
        self.origin_box.clicked.connect(self.toggle_origin)
        
        self.grid_box = QCheckBox("Show Grid")
        self.grid_box.setChecked(False)
        self.grid_box.clicked.connect(self.toggle_grid)

        self.grid_box.setEnabled(False)

        self.axes_tick_box = QCheckBox("Show Axes Ticks")
        self.axes_tick_box.setChecked(False)
        self.axes_tick_box.clicked.connect(self.toggle_axes_ticks)

        self.axes_tick_box.setEnabled(False)

        self.grid_labels_box = QCheckBox("Show Grid Labels")
        self.grid_labels_box.setChecked(False)
        self.grid_labels_box.clicked.connect(self.toggle_grid_labels)

        self.axes_box = QCheckBox("Show Axes")
        self.axes_box.setChecked(False)
        self.axes_box.clicked.connect(self.toggle_axes)

        self.grid_labels_box.setEnabled(False)

        self.anim_speed_layout = QHBoxLayout()

        self.anim_speed_slider_label = QLabel("Animation Speed: ")
        self.anim_speed_slider = QSlider(Qt.Orientation.Horizontal)
        #self.anim_speed_slider.setMinimum(-99) Time reversal
        self.anim_speed_slider_value_label = QLabel("1")
        self.anim_speed_slider.valueChanged.connect(self.anim_speed_func) 

        self.anim_speed_layout.addWidget(self.anim_speed_slider_label)
        self.anim_speed_layout.addWidget(self.anim_speed_slider)
        self.anim_speed_layout.addWidget(self.anim_speed_slider_value_label)

        self.anim_groupbox_layout.addWidget(self.anim_toggle_button)
        self.anim_groupbox_layout.addWidget(self.anim_start_stop_button)
        self.anim_groupbox_layout.addWidget(self.cog_box)
        self.anim_groupbox_layout.addWidget(self.trace_box)
        self.anim_groupbox_layout.addWidget(self.origin_box)
        self.anim_groupbox_layout.addWidget(self.axes_box)
        self.anim_groupbox_layout.addWidget(self.axes_tick_box)
        self.anim_groupbox_layout.addWidget(self.grid_box)
        self.anim_groupbox_layout.addWidget(self.grid_labels_box)
        self.anim_groupbox_layout.addLayout(self.anim_speed_layout)


        self.plot_face_color_layout = QHBoxLayout()

        self.plot_face_color_button = QPushButton("Plot Color")
        self.plot_face_color_preview = QLabel("")

        self.plot_face_color_button.clicked.connect(self.set_plot_face_color)
        self.plot_face_color_layout.addWidget(self.plot_face_color_button)
        self.plot_face_color_layout.addWidget(self.plot_face_color_preview)



        self.anim_groupbox_layout.addLayout(self.plot_face_color_layout)
        

        self.timeLayout = QHBoxLayout()
        self.timeLabel = QLabel("Time: ")
        self.timeValue = QLabel("0")
        self.timeLayout.addWidget(self.timeLabel)
        self.timeLayout.addWidget(self.timeValue)
        self.timeLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.timeLayout.addWidget(self.timeValue)
        self.timeLayout.addWidget(self.timeProgressbar)

        self.rightLayout.addWidget(self.param_groupbox)
        self.rightLayout.addWidget(self.anim_groupbox)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
            
        self.leftLayout.addWidget(self.canvas)
        self.leftWidget = QWidget()
        self.leftWidget.setLayout(self.leftLayout)

        self.leftLayout.addLayout(self.timeLayout)

        self.splitter.addWidget(self.leftWidget)
        
        self.rightWidgetScrollArea = QScrollArea(self.splitter)
        self.rightWidgetScrollArea.setWidgetResizable(True)
        self.rightWidget = QWidget(self.rightWidgetScrollArea)
        self.rightWidget.setLayout(self.rightLayout)
        self.splitter.addWidget(self.rightWidget)

        self.rightWidgetScrollArea.setWidget(self.rightWidget)

        self.layout.addWidget(self.splitter)


    def anim_speed_func(self):
        D = self.anim_speed_slider.value()
        self.anim_speed_slider_value_label.setText(str(D))
        self.anim_speed = D
    
    def set_plot_face_color(self):
        #plt.style.use("fivethirtyeight")
        cd = QColorDialog().getColor()
        self.ax.set_facecolor(cd.name())
        self.plot_face_color_preview.setStyleSheet("background: {}".format(cd.name()))
        self.fig.set_facecolor(cd.name())


    def get_cog_col(self):
        cd = QColorDialog().getColor()
        #self.tb_col1_preview.setText(cd.name())
        self.tb_cog_col = cd.name()
        self.tb_cog_col_preview.setStyleSheet("background: {}".format(cd.name()))

    def get_col1(self):
        cd = QColorDialog().getColor()
        #self.tb_col1_preview.setText(cd.name())
        self.tb_col1 = cd.name()
        self.tb_col1_preview.setStyleSheet("background: {}".format(cd.name()))

    def get_col2(self):
        cd = QColorDialog().getColor()
        #self.tb_col1_preview.setText(cd.name())
        self.tb_col2 = cd.name()
        self.tb_col2_preview.setStyleSheet("background: {}".format(cd.name()))

    def init_vals(self):
        self.get_inputs()
        self.G = 6.6743e-20 # km^3 kg^(-1)s^(-2)

    # Model function
    def TwoBodyProblem(self, y, t, G, m1, m2):
        r1 = y[: 3]
        r2 = y[3 : 6]

        v1 = y[6 : 9]
        v2 = y[9 : 12]

        r = np.linalg.norm(r2 - r1)

        c0 = y[6 : 12]
        c1 = G * m2 * ((r2 - r1)/np.power(r, 3))
        c2 = G * m1 * ((r1 - r2)/np.power(r, 3))

        return np.concatenate((c0, c1, c2))

    def calc(self):
        self.T = len(self.t)
        self.timeProgressbar.setMaximum(2 * int(self.timef.text()))
        y0 = np.concatenate((self.r1, self.r2, self.v1, self.v2))
        y = odeint(self.TwoBodyProblem, y0, self.t, args=(self.G, self.m1, self.m2))

        self.r1_sol = y[:, :3]
        self.r2_sol = y[:, 3: 6]

        # Center of mass
        self.cog_sol = (self.m1 * self.r1_sol + self.m2 * self.r2_sol)/(self.m1 + self.m2)



    

if __name__ == "__main__":
    qapp = QApplication(sys.argv)
    window = MainWindow(objectName="MainWindow")
    window.show()
    sys.exit(qapp.exec())
