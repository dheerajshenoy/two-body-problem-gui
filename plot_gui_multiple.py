import sys
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, QLineEdit, QGroupBox, QGridLayout, QLabel, QColorDialog, QSplitter, QComboBox, QMenuBar, QMenu, QSizePolicy, QProgressBar, QWidget, QMainWindow, QScrollArea, QSlider, QMessageBox, QFrame
from PyQt6.QtGui import QAction, QShortcut, QKeySequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation, FuncAnimation
from scipy.integrate import odeint

import numpy as np

app_stylesheet = '''

#TimeProgressBar {
        }

QWidget {
        font-family: Ubuntu;
        font-size: 18px;
        }
'''

msgbox_stylesheet = '''
font-family: Rajdhani Semibold;
font-size: 24px;

'''

line_stylesheet = """
padding: 20px;
background-color: gray;
"""

class ToolBar(QWidget):
    def __init__(self, parent = None, **kwargs):
        super(ToolBar, self).__init__(parent, **kwargs)
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

    def addWidget(self, widget):
        self.layout.addWidget(widget)
    
    def addLayout(self, layout):
        self.layout.addLayout(layout)

class Line(QFrame):
    def __init__(self, parent = None, objectName = "Line", **kwargs):
        super(Line, self).__init__(parent, **kwargs)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet(line_stylesheet)
        self.setFrameShadow(QFrame.Shadow.Sunken)

class PreferencesDialog(QMainWindow):
    def __init__(self, parent=None):
        super(PreferencesDialog, self).__init__(parent)
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.setCentralWidget(self.widget)

class MainWindow(QMainWindow):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)

        self.num = 0
        self.paused = False
        self.cog_shown = True
        self.anim_start = False
        self.trace_shown = True
        self.origin_shown = True
        self.axes_shown = True
        self.axes_ticks_shown = False
        self.grid_shown = False
        self.grid_labels_shown = False
        self.energy_plot_shown = False
        self.tb_col1 = "#FF5000"
        self.tb_col2 = "#563843"
        self.tb_col3 = "#456753"
        self.tb_cog_col = "#342482"
        self.plot_bg_color = "#989898"
        self.time = 0
        self.anim_speed = 10
        self.three_body_mode = False

        self.setMinimumSize(800, 400)
        self.setStyleSheet(app_stylesheet)

        self._main = QWidget()
        self.setCentralWidget(self._main)
        self.layout = QVBoxLayout(self._main)

        self.toolbar = ToolBar()
        
        self.plotInit()


        # self.ax.w_xaxis.set_pane_color((0.75, 0.5, 0.3, 1))
        # self.ax.w_yaxis.set_pane_color((R,G,B,A))
        # self.ax.w_zaxis.set_pane_color((R,G,B,A))

        self.guiInit()
        

        self.tb_col1_preview.setStyleSheet("background: {}".format(self.tb_col1))
        self.tb_col2_preview.setStyleSheet("background: {}".format(self.tb_col2))
        self.tb_col3_preview.setStyleSheet("background: {}".format(self.tb_col3))
        self.tb_cog_col_preview.setStyleSheet("background: {}".format(self.tb_cog_col))
        self.plot_face_color_preview.setStyleSheet("background: {}".format(self.plot_bg_color))

        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        self.ax.set_axis_off()
        self.ax.grid(False)
        self.ax.set_title("Non-inertial frame of reference")

        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.set_zticklabels([])
        self.ax2.set_axis_off()
        self.ax2.grid(False)
        self.ax2.set_title("Center of Gravity frame of reference", x = .7, y = -0.1)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.animate_func)

        self.timer_three_body = QtCore.QTimer()
        self.timer_three_body.setInterval(10)
        self.timer_three_body.timeout.connect(self.animate_three_body_func)


        self.init_vals()
        self.calc()


        self.ax3.set_xlim(0, self.xmax)
        self.ax4.set_xlim(0, self.xmax)
        self.ax5.set_xlim(0, self.xmax)

        self.ax4.set_ylim(self.ymin2, self.ymax2)
        self.ax3.set_ylim(self.ymin1, self.ymax1)
        self.ax5.set_ylim(self.ymin3, self.ymax3)


        self.show()

    def plotInit(self):
        self.fig = Figure(figsize=(10, 10), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        self.ax = self.fig.add_subplot(331, projection='3d')
        self.ax2 = self.fig.add_subplot(332, projection='3d')
        self.ax3 = self.fig.add_subplot(333)
        self.ax4 = self.fig.add_subplot(334)
        self.ax5 = self.fig.add_subplot(335)
        
        self.plotList = [self.ax, self.ax2, self.ax3, self.ax4, self.ax5]
        self.energyPlotAxesList = [self.ax3, self.ax4, self.ax5]

        self.ax.set_aspect('equal', 'box')
        self.ax2.set_aspect('equal', 'box')
        #self.ax3.set_aspect('equal', 'box')
        
        self.ax.set_position([-0.12,0.25,0.8,0.8])
        self.ax2.set_position([-0.01, 0.02, 0.3, 0.3])

        self.ax3.set_position([0.7, 0.4, 0.25, 0.25])
        self.ax4.set_position([0.7, 0.7, 0.25, 0.25])
        self.ax5.set_position([0.7, 0.1, 0.25, 0.25])
        
        for i in self.plotList:
            i.set_facecolor(self.plot_bg_color)

        if not self.energy_plot_shown:
            for i in self.energyPlotAxesList:
                i.set_visible(False)
            self.ax.set_position([0.1,0.25,0.8,0.8])
            self.ax2.set_position([-0.01, 0.02, 0.3, 0.3])
            

        self.fig.set_facecolor(self.plot_bg_color)

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

        self.canvas.draw()


    def toggle_axes_ticks(self):
        self.axes_ticks_shown = not self.axes_ticks_shown
        if self.axes_ticks_shown:
            self.grid_box.setEnabled(True)
            self.grid_labels_box.setEnabled(True)
        else:
            self.grid_box.setEnabled(False)
            self.grid_labels_box.setEnabled(False)

        self.canvas.draw()

    def toggle_grid(self):
        self.grid_shown = not self.grid_shown
        if self.grid_shown:
            self.grid_labels_box.setEnabled(True)
        else:
            self.grid_labels_box.setEnabled(False)
        self.canvas.draw()

    def toggle_grid_labels(self):
        self.grid_labels_shown = not self.grid_labels_shown
        self.canvas.draw()

    def anim_toggle(self):
        self.paused = not self.paused
        if self.paused:
            self.anim_toggle_button.setText("Play Animation")
        else:
            self.anim_toggle_button.setText("Pause Animation")
        self.canvas.draw()

    def animate_func(self):
        if not self.paused:
            if self.num + self.anim_speed < self.T - 1:
                self.num += self.anim_speed
            else:
                self.num = 0

            self.timeProgressbar.setValue(abs(self.num))
            self.timeValue.setText("{} s".format(str(self.t[self.num])))
            self.ax.cla()
            self.ax2.cla()

            if self.energy_plot_shown:
                self.ax3.cla()
                self.ax4.cla()
                self.ax5.cla()
                self.ax3.plot(self.KE1[: self.num + 1],'r', markersize=1)
                self.ax4.plot(self.KE2[: self.num + 1],'b', markersize=1)
                self.ax5.plot(self.totalE[: self.num + 1],'g', markersize=1)

            # Energy plot
            # self.ax3.plot(self.num, self.v1_res[self.num],'.b')
            # self.ax3.plot(self.num, self.v2_res[self.num],'.r')

            if self.cog_shown:
                self.ax.scatter(self.cog_sol[self.num, 0], self.cog_sol[self.num, 1], self.cog_sol[self.num, 2], c=self.tb_cog_col, marker='o', s=self.radius_cog)
                self.ax2.scatter(self.cog_sol_t[self.num, 0], self.cog_sol_t[self.num, 1], self.cog_sol_t[self.num, 2], c=self.tb_cog_col, marker='o', s=self.radius_cog)

            if self.trace_shown:
                self.ax.plot3D(self.r1_sol[: self.num + 1, 0], self.r1_sol[: self.num + 1, 1], self.r1_sol[: self.num + 1, 2], c = self.tb_col1)
                self.ax.plot3D(self.r2_sol[: self.num + 1, 0], self.r2_sol[: self.num + 1, 1], self.r2_sol[: self.num + 1, 2], c = self.tb_col2)

                self.ax2.plot3D(self.t1_sol[: self.num + 1, 0], self.t1_sol[: self.num + 1, 1], self.t1_sol[: self.num + 1, 2], c = self.tb_col1)
                self.ax2.plot3D(self.t2_sol[: self.num + 1, 0], self.t2_sol[: self.num + 1, 1], self.t2_sol[: self.num + 1, 2], c = self.tb_col2)

            self.ax.scatter(self.r1_sol[self.num, 0], self.r1_sol[self.num, 1], self.r1_sol[self.num, 2], c= self.tb_col1, marker='o', s=self.radius1)
            self.ax.scatter(self.r2_sol[self.num, 0], self.r2_sol[self.num, 1], self.r2_sol[self.num, 2], c=self.tb_col2, marker='o', s=self.radius2)

            self.ax2.scatter(self.t1_sol[self.num, 0], self.t1_sol[self.num, 1], self.t1_sol[self.num, 2], c= self.tb_col1, marker='o', s=self.radius1)
            self.ax2.scatter(self.t2_sol[self.num, 0], self.t2_sol[self.num, 1], self.t2_sol[self.num, 2], c=self.tb_col2, marker='o', s=self.radius2)

            if self.origin_shown:
                self.ax.scatter(self.r1_sol[0, 0], self.r1_sol[0, 1], self.r1_sol[0, 2], c='black', marker='o', s=50)
                self.ax.scatter(self.r2_sol[0, 0], self.r2_sol[0, 1], self.r2_sol[0, 2], c='black', marker='o', s=50)

            if not self.grid_shown:
                self.ax.grid(False)
                self.ax2.grid(False)

            if not self.axes_shown:
                self.ax.set_axis_off()
                self.ax2.set_axis_off()

            if not self.grid_labels_shown:
                self.ax.set_xticklabels([])
                self.ax.set_yticklabels([])
                self.ax.set_zticklabels([])
                self.ax2.set_xticklabels([])
                self.ax2.set_yticklabels([])
                self.ax2.set_zticklabels([])

            if not self.axes_ticks_shown:
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_zticks([])

                self.ax2.set_xticks([])
                self.ax2.set_yticks([])
                self.ax2.set_zticks([])


            self.canvas.draw()

    def animate_three_body_func(self):
        if not self.paused:
            if self.num + self.anim_speed < self.T - 1:
                self.num += self.anim_speed
            else:
                self.num = 0

            self.timeProgressbar.setValue(abs(self.num))
            self.timeValue.setText("{} s".format(str(self.t[self.num])))
            self.ax.cla()
            self.ax2.cla()
            

            if self.cog_shown:
                self.ax.scatter(self.cog_sol[self.num, 0], self.cog_sol[self.num, 1], self.cog_sol[self.num, 2], c=self.tb_cog_col, marker='o', s=self.radius_cog)
                self.ax2.scatter(self.cog_sol_t[self.num, 0], self.cog_sol_t[self.num, 1], self.cog_sol_t[self.num, 2], c=self.tb_cog_col, marker='o', s=self.radius_cog)

            if self.trace_shown:
                self.ax.plot3D(self.r1_sol[: self.num + 1, 0], self.r1_sol[: self.num + 1, 1], self.r1_sol[: self.num + 1, 2], c = self.tb_col1)
                self.ax.plot3D(self.r2_sol[: self.num + 1, 0], self.r2_sol[: self.num + 1, 1], self.r2_sol[: self.num + 1, 2], c = self.tb_col2)
                self.ax.plot3D(self.r3_sol[: self.num + 1, 0], self.r3_sol[: self.num + 1, 1], self.r3_sol[: self.num + 1, 2], c = self.tb_col3)

                self.ax2.plot3D(self.t1_sol[: self.num + 1, 0], self.t1_sol[: self.num + 1, 1], self.t1_sol[: self.num + 1, 2], c = self.tb_col1)
                self.ax2.plot3D(self.t2_sol[: self.num + 1, 0], self.t2_sol[: self.num + 1, 1], self.t2_sol[: self.num + 1, 2], c = self.tb_col2)
                self.ax2.plot3D(self.t3_sol[: self.num + 1, 0], self.t3_sol[: self.num + 1, 1], self.t3_sol[: self.num + 1, 2], c = self.tb_col3)

            self.ax.set_title("Non-inertial frame of reference")
            self.ax.scatter(self.r1_sol[self.num, 0], self.r1_sol[self.num, 1], self.r1_sol[self.num, 2], c= self.tb_col1, marker='o', s=self.radius1)
            self.ax.scatter(self.r2_sol[self.num, 0], self.r2_sol[self.num, 1], self.r2_sol[self.num, 2], c=self.tb_col2, marker='o', s=self.radius2)
            self.ax.scatter(self.r3_sol[self.num, 0], self.r3_sol[self.num, 1], self.r3_sol[self.num, 2], c=self.tb_col3, marker='o', s=self.radius3)

            self.ax2.scatter(self.t1_sol[self.num, 0], self.t1_sol[self.num, 1], self.t1_sol[self.num, 2], c= self.tb_col1, marker='o', s=self.radius1)
            self.ax2.scatter(self.t2_sol[self.num, 0], self.t2_sol[self.num, 1], self.t2_sol[self.num, 2], c=self.tb_col2, marker='o', s=self.radius2)
            self.ax2.scatter(self.t3_sol[self.num, 0], self.t3_sol[self.num, 1], self.t3_sol[self.num, 2], c=self.tb_col3, marker='o', s=self.radius3)
            self.ax2.set_title("Center of Gravity frame of reference", x = .7, y = -0.1)

            if self.origin_shown:
                self.ax.scatter(self.r1_sol[0, 0], self.r1_sol[0, 1], self.r1_sol[0, 2], c='black', marker='o', s=50)
                self.ax.scatter(self.r2_sol[0, 0], self.r2_sol[0, 1], self.r2_sol[0, 2], c='black', marker='o', s=50)
                self.ax.scatter(self.r3_sol[0, 0], self.r3_sol[0, 1], self.r3_sol[0, 2], c='black', marker='o', s=50)

            if not self.grid_shown:
                self.ax.grid(False)
                self.ax2.grid(False)

            if not self.axes_shown:
                self.ax.set_axis_off()
                self.ax2.set_axis_off()

            if not self.grid_labels_shown:
                self.ax.set_xticklabels([])
                self.ax.set_yticklabels([])
                self.ax.set_zticklabels([])
                self.ax2.set_xticklabels([])
                self.ax2.set_yticklabels([])
                self.ax2.set_zticklabels([])

            if not self.axes_ticks_shown:
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_zticks([])

                self.ax2.set_xticks([])
                self.ax2.set_yticks([])
                self.ax2.set_zticks([])

            self.canvas.draw()

    # Function for starting and stopping animation
    def anim_start_stop(self):
        self.anim_start_stop_button.setText("Reset Animation")
        self.get_inputs()
        self.calc()
        if not self.three_body_mode:
            self.timer.start()
        else:
            self.timer_three_body.start()
        self.num = 0

        self.ax.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()

        if self.paused:
            if not self.grid_shown:
                self.ax.grid(False)
                self.ax2.grid(False)

            if not self.axes_shown:
                self.ax.set_axis_off()
                self.ax2.set_axis_off()

            if not self.grid_labels_shown:
                self.ax.set_xticklabels([])
                self.ax.set_yticklabels([])
                self.ax.set_zticklabels([])
                self.ax2.set_xticklabels([])
                self.ax2.set_yticklabels([])
                self.ax2.set_zticklabels([])

            if not self.axes_ticks_shown:
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_zticks([])

                self.ax2.set_xticks([])
                self.ax2.set_yticks([])
                self.ax2.set_zticks([])

            self.canvas.draw()



    # Function for getting the inputs from lineedits
    def get_inputs(self):
        try:
            self.m1 = float(self.tb_m1.text())
            self.r1 = np.array([float(self.tb_r1x.text()), float(self.tb_r1y.text()), float(self.tb_r1z.text())])
            self.v1 = np.array([float(self.tb_v1x.text()), float(self.tb_v1y.text()), float(self.tb_v1z.text())])
            self.radius1 = float(self.tb_radius1.text()) * 100

            self.m2 = float(self.tb_m2.text())
            self.r2 = np.array([float(self.tb_r2x.text()), float(self.tb_r2y.text()), float(self.tb_r2z.text())])
            self.v2 = np.array([float(self.tb_v2x.text()), float(self.tb_v2y.text()), float(self.tb_v2z.text())])
            self.radius2 = float(self.tb_radius2.text()) * 100

            if self.three_body_mode:
                self.m3 = float(self.tb_m3.text())
                self.r3 = np.array([float(self.tb_r3x.text()), float(self.tb_r3y.text()), float(self.tb_r3z.text())])
                self.v3 = np.array([float(self.tb_v3x.text()), float(self.tb_v3y.text()), float(self.tb_v3z.text())])
                self.radius3 = float(self.tb_radius3.text()) * 100

            self.radius_cog = float(self.tb_radius_cog.text()) * 100

            self.t = np.arange(float(self.time0.text()), float(self.timef.text()), float(self.timedt.text()))

            self.xmax = len(self.t)


            # self.ax3.set_xlim(0, self.xmax)
            # self.ax4.set_xlim(0, self.xmax)
            # self.ax5.set_xlim(0, self.xmax)



        except ValueError:
            msg = QMessageBox(self)
            msg.setStyleSheet(msgbox_stylesheet)
            msg.setText("Please check the values entered")
            msg.show()


    def initMenu(self):
        self.menubar = QMenuBar()

        self.edit_menu = QMenu("&Edit", self.menubar)
        self.view_menu = QMenu("&View", self.menubar)
        self.about_menu = QAction("&About", self.menubar)

        self.menubar.addMenu(self.edit_menu)
        self.menubar.addMenu(self.view_menu)
        self.menubar.addAction(self.about_menu)

        self.about_menu.triggered.connect(self.show_about)

        self.prefs = QAction("Preferences", self)
        self.edit_menu.addAction(self.prefs)

        self.prefs.triggered.connect(self.show_prefs_dialog)

        self.view_sidebar = QAction("Sidebar", self, checkable = True)
        self.view_sidebar.setChecked(True)

        self.view_sidebar.triggered.connect(self.toggle_sidebar)


        self.view_menu.addAction(self.view_sidebar)

        self.view_plot_menu = QMenu("Plot", self.view_menu)
        self.view_non_inertial_frame = QAction("Non-inertial frame", self, checkable = True)
        self.view_non_inertial_frame.triggered.connect(self.view_non_inertial_func)
        self.view_non_inertial_frame.setChecked(True)
        self.view_cog_frame = QAction("COG frame", self, checkable = True)
        self.view_cog_frame.triggered.connect(self.view_cog_func)
        self.view_cog_frame.setChecked(True)
        self.view_plot_menu.addAction(self.view_non_inertial_frame)
        self.view_plot_menu.addAction(self.view_cog_frame)

        self.view_energy_action = QAction("Energy", self, checkable = True)
        self.view_energy_action.triggered.connect(self.view_energy_func)
        self.view_energy_action.setChecked(self.energy_plot_shown)

        self.view_plot_menu.addAction(self.view_energy_action)

        self.view_menu.addMenu(self.view_plot_menu)

        self.setMenuBar(self.menubar)

    def view_energy_func(self):
        self.energy_plot_shown = not self.energy_plot_shown
        if self.energy_plot_shown:
            for i in self.energyPlotAxesList:
                i.set_visible(True)
        
            self.ax.set_position([-0.12,0.25,0.8,0.8])
            self.ax2.set_position([-0.01, 0.02, 0.3, 0.3])
        else:
            for i in self.energyPlotAxesList:
                i.set_visible(False)
        
            self.ax.set_position([0.1,0.25,0.8,0.8])
            self.ax2.set_position([-0.01, 0.02, 0.3, 0.3])

        self.canvas.draw()

    def show_prefs_dialog(self):
        prefs = PreferencesDialog(self)
        prefs.show()

    def show_about(self):
        msg = QMessageBox(self)
        msg.setStyleSheet(msgbox_stylesheet)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("Two and three body simulation program written in Python.\n\nCode by: V DHEERAJ SHENOY")
        msg.show()

    def toggle_sidebar(self):
        self.rightWidgetScrollArea.setHidden(not self.rightWidgetScrollArea.isHidden())

    def view_non_inertial_func(self):
        self.ax.set_visible(self.view_non_inertial_frame.isChecked())
        self.canvas.draw()

    def view_cog_func(self):
        self.ax2.set_visible(self.view_cog_frame.isChecked())
        self.canvas.draw()

    # Function for GUI Components
    def guiInit(self):

        self.initMenu()

        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()

        # mass 0.330,4.87,5.97,0.073,0.642,1898,568,86.8,102,0.0130

        self.timeProgressbar = QProgressBar(objectName="TimeProgressBar")

        # Group Box

        self.three_body_toggle_checkbox = QCheckBox("Three body problem mode")
        self.three_body_toggle_checkbox.clicked.connect(self.three_body_toggle_func)

        self.toolbar.addWidget(self.three_body_toggle_checkbox)

        self.anim_groupbox = QGroupBox("Animation")
        self.anim_groupbox_layout = QVBoxLayout(self.anim_groupbox)

        self.massPresetLayout = QHBoxLayout()

        self.param_groupbox = QGroupBox("Parameters")
        self.param_groupbox_layout = QGridLayout(self.param_groupbox)

        # Parameter GroupBox Elements

        self.tb_m1 = QLineEdit("1e26")
        self.tb_v1_layout = QHBoxLayout()
        self.tb_r1_layout = QHBoxLayout()

        self.tb_m2 = QLineEdit("1e20")
        self.tb_v2_layout = QHBoxLayout()
        self.tb_r2_layout = QHBoxLayout()

        self.tb_m3 = QLineEdit("1e10")
        self.tb_v3_layout = QHBoxLayout()
        self.tb_r3_layout = QHBoxLayout()

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

        self.tb_r3x = QLineEdit("3000")
        self.tb_r3y = QLineEdit("0")
        self.tb_r3z = QLineEdit("0")

        self.tb_v3x = QLineEdit("0")
        self.tb_v3y = QLineEdit("40")
        self.tb_v3z = QLineEdit("0")

        self.tb_r3_layout.addWidget(self.tb_r3x)
        self.tb_r3_layout.addWidget(self.tb_r3y)
        self.tb_r3_layout.addWidget(self.tb_r3z)

        self.tb_v3_layout.addWidget(self.tb_v3x)
        self.tb_v3_layout.addWidget(self.tb_v3y)
        self.tb_v3_layout.addWidget(self.tb_v3z)

        self.tb_col1_button = QPushButton("Color 1")
        self.tb_col1_button.clicked.connect(self.get_col1)
        self.tb_col1_preview = QLabel(" ")

        self.tb_col2_button = QPushButton("Color 2")
        self.tb_col2_button.clicked.connect(self.get_col2)
        self.tb_col2_preview = QLabel(" ")

        self.tb_col3_button = QPushButton("Color 3")
        self.tb_col3_button.clicked.connect(self.get_col3)
        self.tb_col3_preview = QLabel(" ")

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

        self.m3_label = QLabel("Mass 3")
        self.tb_m3 = QLineEdit("1e10")

        self.pos3_label = QLabel("Position 3")
        self.v3_label = QLabel("Velocity 3")

        self.m3_label.setHidden(not self.three_body_mode)
        self.tb_m3.setHidden(not self.three_body_mode)
        self.pos3_label.setHidden(not self.three_body_mode)
        self.v3_label.setHidden(not self.three_body_mode)
        self.tb_col3_preview.setHidden(not self.three_body_mode)
        self.tb_col3_button.setHidden(not self.three_body_mode)

        self.sep_2 = Line()
        self.sep_2.setHidden(not self.three_body_mode)

        for i in range(3):
            self.tb_r3_layout.itemAt(i).widget().setHidden(not self.three_body_mode)
            self.tb_v3_layout.itemAt(i).widget().setHidden(not self.three_body_mode)

        self.tb_radius1 = QLineEdit("2")
        self.tb_radius2 = QLineEdit("2")
        self.tb_radius3 = QLineEdit("2")
        self.tb_radius3.setHidden(not self.three_body_mode)

        self.tb_radius_cog = QLineEdit("1")

        self.radius3_label = QLabel("Radius 3")
        self.radius3_label.setHidden(not self.three_body_mode)

        self.param_groupbox_layout.addWidget(QLabel("Mass 1 "), 0, 0)
        self.param_groupbox_layout.addWidget(self.tb_m1, 0, 1)

        self.param_groupbox_layout.addWidget(QLabel("Position 1"), 1, 0)
        self.param_groupbox_layout.addLayout(self.tb_r1_layout, 1, 1)

        self.param_groupbox_layout.addWidget(QLabel("Velocity 1"), 2, 0)
        self.param_groupbox_layout.addLayout(self.tb_v1_layout, 2, 1)

        self.param_groupbox_layout.addWidget(self.tb_col1_button, 3, 0)
        self.param_groupbox_layout.addWidget(self.tb_col1_preview, 3, 1)

        self.param_groupbox_layout.addWidget(QLabel("Radius 1"), 4, 0)
        self.param_groupbox_layout.addWidget(self.tb_radius1, 4, 1)

        self.param_groupbox_layout.addWidget(Line(), 5, 0, 1, 2)

        self.param_groupbox_layout.addWidget(QLabel("Mass 2 "), 6, 0)
        self.param_groupbox_layout.addWidget(self.tb_m2, 6, 1)

        self.param_groupbox_layout.addWidget(QLabel("Position 2"), 7, 0)
        self.param_groupbox_layout.addLayout(self.tb_r2_layout, 7, 1)

        self.param_groupbox_layout.addWidget(QLabel("Velocity 2"), 8, 0)
        self.param_groupbox_layout.addLayout(self.tb_v2_layout, 8, 1)

        self.param_groupbox_layout.addWidget(self.tb_col2_button, 9, 0)
        self.param_groupbox_layout.addWidget(self.tb_col2_preview, 9, 1)

        self.param_groupbox_layout.addWidget(QLabel("Radius 2"), 10, 0)
        self.param_groupbox_layout.addWidget(self.tb_radius2, 10, 1)

        self.param_groupbox_layout.addWidget(self.sep_2, 11, 0, 1, 2)

        self.param_groupbox_layout.addWidget(self.m3_label, 12, 0)
        self.param_groupbox_layout.addWidget(self.tb_m3, 12, 1)

        self.param_groupbox_layout.addWidget(self.pos3_label, 13, 0)
        self.param_groupbox_layout.addLayout(self.tb_r3_layout, 13, 1)

        self.param_groupbox_layout.addWidget(self.v3_label, 14, 0)
        self.param_groupbox_layout.addLayout(self.tb_v3_layout, 14, 1)


        self.param_groupbox_layout.addWidget(self.tb_col3_button, 15, 0)
        self.param_groupbox_layout.addWidget(self.tb_col3_preview, 15, 1)


        self.param_groupbox_layout.addWidget(self.radius3_label, 16, 0)
        self.param_groupbox_layout.addWidget(self.tb_radius3, 16, 1)

        self.param_groupbox_layout.addWidget(Line(), 17, 0, 1, 2)

        self.param_groupbox_layout.addWidget(self.tb_cog_col_button, 18, 0)
        self.param_groupbox_layout.addWidget(self.tb_cog_col_preview, 18, 1)

        self.param_groupbox_layout.addWidget(QLabel("Time"), 19, 0)
        self.param_groupbox_layout.addLayout(self.time_layout,19, 1)

        self.param_groupbox_layout.addWidget(QLabel("Radius COG"), 20, 0)
        self.param_groupbox_layout.addWidget(self.tb_radius_cog, 20, 1)

        # Buttons

        self.anim_layout = QHBoxLayout()
        
        self.anim_start_stop_button = QPushButton("Start Animation")
        self.anim_start_stop_button.setStyleSheet("padding-left: 10px; padding-right: 10px;")
        self.anim_start_stop_button.clicked.connect(self.anim_start_stop)

        self.anim_toggle_button = QPushButton("Pause Animation")
        self.anim_toggle_button.setStyleSheet("padding-left: 10px; padding-right: 10px;")
        self.anim_toggle_button.clicked.connect(self.anim_toggle)

        self.anim_layout.addWidget(self.anim_toggle_button)
        self.anim_layout.addWidget(self.anim_start_stop_button)

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

        self.grid_box.setEnabled(self.grid_shown)

        self.axes_tick_box = QCheckBox("Show Axes Ticks")
        self.axes_tick_box.setChecked(self.axes_ticks_shown)
        self.axes_tick_box.setEnabled(self.grid_shown)
        self.axes_tick_box.clicked.connect(self.toggle_axes_ticks)

        self.axes_tick_box.setEnabled(self.axes_ticks_shown)

        self.grid_labels_box = QCheckBox("Show Grid Labels")
        self.grid_labels_box.setChecked(self.grid_labels_shown)
        self.grid_labels_box.clicked.connect(self.toggle_grid_labels)

        self.axes_box = QCheckBox("Show Axes")
        self.axes_box.setChecked(self.axes_shown)
        self.axes_box.clicked.connect(self.toggle_axes)

        self.grid_labels_box.setEnabled(self.grid_labels_shown)

        self.anim_speed_layout = QHBoxLayout()

        self.anim_speed_slider_label = QLabel("Animation Speed: ")
        self.anim_speed_slider = QSlider(Qt.Orientation.Horizontal)
        #self.anim_speed_slider.setMinimum(-99) Time reversal
        self.anim_speed_slider_value_label = QLabel("1")
        self.anim_speed_slider.valueChanged.connect(self.anim_speed_func) 

        self.anim_speed_layout.addWidget(self.anim_speed_slider_label)
        self.anim_speed_layout.addWidget(self.anim_speed_slider)
        self.anim_speed_layout.addWidget(self.anim_speed_slider_value_label)
        self.anim_groupbox_layout.addWidget(self.cog_box)
        self.anim_groupbox_layout.addWidget(self.trace_box)
        self.anim_groupbox_layout.addWidget(self.origin_box)
        self.anim_groupbox_layout.addWidget(self.axes_box)
        self.anim_groupbox_layout.addWidget(self.axes_tick_box)
        self.anim_groupbox_layout.addWidget(self.grid_box)
        self.anim_groupbox_layout.addWidget(self.grid_labels_box)
        self.toolbar.addLayout(self.anim_speed_layout)
        self.toolbar.addLayout(self.massPresetLayout)
        self.toolbar.addLayout(self.anim_layout)


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
        self.leftWidget.setContentsMargins(0, 0, 0, 0)
        self.leftLayout.setContentsMargins(0, 0, 0, 0)
        self.rightLayout.setContentsMargins(0, 0, 0, 0)
        self._main.setContentsMargins(0, 0, 0, 0)

        self.leftLayout.addLayout(self.timeLayout)

        self.splitter.addWidget(self.leftWidget)

        self.rightWidgetScrollArea = QScrollArea(self.splitter)
        self.rightWidgetScrollArea.setWidgetResizable(True)
        self.rightWidget = QWidget(self.rightWidgetScrollArea)
        self.rightWidget.setLayout(self.rightLayout)
        self.splitter.addWidget(self.rightWidget)

        self.rightWidget.setContentsMargins(0, 0, 0, 0)
        self.rightWidgetScrollArea.setWidget(self.rightWidget)
        
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.splitter)

    def three_body_toggle_func(self):
        self.three_body_mode = not self.three_body_mode

        self.radius3_label.setHidden(not self.three_body_mode)
        self.tb_radius3.setHidden(not self.three_body_mode)
        self.sep_2.setHidden(not self.three_body_mode)
        self.m3_label.setHidden(not self.three_body_mode)
        self.tb_m3.setHidden(not self.three_body_mode)

        self.v3_label.setHidden(not self.three_body_mode)
        self.pos3_label.setHidden(not self.three_body_mode)

        self.tb_col3_button.setHidden(not self.three_body_mode)
        self.tb_col3_preview.setHidden(not self.three_body_mode)

        for i in range(3):
            self.tb_r3_layout.itemAt(i).widget().setHidden(not self.three_body_mode)
            self.tb_v3_layout.itemAt(i).widget().setHidden(not self.three_body_mode)

        self.pause_animation(True)

        if self.three_body_mode:
            self.timer.stop()
        else:
            self.timer_three_body.stop()

    def pause_animation(self, bool):
        self.paused = bool
        if self.paused:
            self.anim_toggle_button.setText("Play Animation")
        else:
            self.anim_toggle_button.setText("Pause Animation")

    def anim_speed_func(self):
        D = self.anim_speed_slider.value()
        self.anim_speed_slider_value_label.setText(str(D))
        self.anim_speed = D

    def set_plot_face_color(self):
        #plt.style.use("fivethirtyeight")
        cd = QColorDialog().getColor()

        self.ax.set_facecolor(cd.name())
        self.ax2.set_facecolor(cd.name())
        self.ax3.set_facecolor(cd.name())
        self.ax4.set_facecolor(cd.name())
        self.ax5.set_facecolor(cd.name())

        self.plot_face_color_preview.setStyleSheet("background: {}".format(cd.name()))
        self.fig.set_facecolor(cd.name())
        self.canvas.draw()


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

    def get_col3(self):
        cd = QColorDialog().getColor()
        #self.tb_col1_preview.setText(cd.name())
        self.tb_col3 = cd.name()
        self.tb_col3_preview.setStyleSheet("background: {}".format(cd.name()))

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

    def ThreeBodyProblem(self, y, t, G, m1, m2, m3):
        r1 = y[: 3]
        r2 = y[3 : 6]
        r3 = y[6 : 9]

        v1 = y[9: 12]
        v2 = y[12 : 15]
        v3 = y[15: 18]

        r12 = np.linalg.norm(r2 - r1)
        r13 = np.linalg.norm(r1 - r3)
        r23 = np.linalg.norm(r2 - r3)

        c0 = y[9 : 18]
        c1 = G * (m2 * ((r2 - r1)/np.power(r12, 3)) + m3 * ((r3 - r1)/np.power(r13, 3)))
        c2 = G * (m1 * ((r1 - r2)/np.power(r12, 3)) + m3 * ((r3 - r2)/np.power(r23, 3)))
        c3 = G * (m1 * ((r1 - r3)/np.power(r13, 3)) + m2 * ((r2 - r3)/np.power(r23, 3)))

        return np.concatenate((c0, c1, c2, c3))

    def calc(self):
        self.T = len(self.t)
        self.timeProgressbar.setMaximum(len(self.t))

        if not self.three_body_mode:
            y0 = np.concatenate((self.r1, self.r2, self.v1, self.v2))
            y = odeint(self.TwoBodyProblem, y0, self.t, args=(self.G, self.m1, self.m2))

            self.r1_sol = y[:, :3]
            self.r2_sol = y[:, 3: 6]
            
            self.v1_sol = y[:, 6: 9]
            self.v2_sol = y[:, 9: 12]

            self.v1_res = [round(np.linalg.norm(i), 6) for i in self.v1_sol]
            self.v2_res = [round(np.linalg.norm(i), 6) for i in self.v2_sol]

            self.KE1 = 0.5 * self.m1 * np.array([np.power(i, 2) for i in self.v1_res])
            self.KE2 = 0.5 * self.m1 * np.array([np.power(i, 2) for i in self.v2_res])

            self.totalE = [x + y for x, y in zip(self.KE1, self.KE2)] 

            self.ymin1, self.ymax1 = min(self.KE1), max(self.KE1)
            self.ymin2, self.ymax2 = min(self.KE2), max(self.KE2)
            self.ymin3, self.ymax3 = min(self.totalE), max(self.totalE)
            
            # Center of mass
            self.cog_sol = (self.m1 * self.r1_sol + self.m2 * self.r2_sol)/(self.m1 + self.m2)

            self.t1_sol = self.r1_sol - self.cog_sol
            self.t2_sol = self.r2_sol - self.cog_sol

            self.cog_sol_t = (self.m1 * self.t1_sol + self.m2 * self.t2_sol)/(self.m1 + self.m2)
        else:
            self.get_inputs()
            y0 = np.concatenate((self.r1, self.r2, self.r3, self.v1, self.v2, self.v3))
            y = odeint(self.ThreeBodyProblem, y0, self.t, args=(self.G, self.m1, self.m2, self.m3))

            self.r1_sol = y[:, :3]
            self.r2_sol = y[:, 3: 6]
            self.r3_sol = y[:, 6: 9]

            # Center of mass
            self.cog_sol = (self.m1 * self.r1_sol + self.m2 * self.r2_sol + self.m3 * self.r3_sol)/(self.m1 + self.m2 + self.m3)

            self.t1_sol = self.r1_sol - self.cog_sol
            self.t2_sol = self.r2_sol - self.cog_sol
            self.t3_sol = self.r3_sol - self.cog_sol

            self.cog_sol_t = (self.m1 * self.t1_sol + self.m2 * self.t2_sol + self.m3 * self.t3_sol)/(self.m1 + self.m2 + self.m3)

if __name__ == "__main__":
    qapp = QApplication(sys.argv)
    window = MainWindow(objectName="MainWindow")
    window.show()
    sys.exit(qapp.exec())
