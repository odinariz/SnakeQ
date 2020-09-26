import numpy as np
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt

import parameters as p
import environment
import snake_sensors
import DQNAgent


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.top, self.left, self.width, self.height = p.MainWidgetSize
    
    def initUI(self):
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('SnakeQ by ludius0')
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Create Game window
        self.topEnv, self.leftEnv, self.widthEnv, self.heightEnv = p.EnvWidgetSize
        self.envWidget = EnvWidget(self.centralWidget)
        self.snake_widget_window.setGeometry(QtCore.QRect(self.topEnv+10, self.leftEnv+10, self.widthEnv, self.heightEnv))

class EnvWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        pass

class VisualiseSensorsWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        # For distance to wall will be 4 rects which will light up from 0 to 1
        # for apple and own body sensors will be 8 rects which will immediately light up if True
        pass

class Statistics(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        # count generations ect...
        pass

if __name__ == "__main__":
    net = DQNAgent.Neural_Network()
    agent = DQNAgent.QAgent(net)
    env = environment.Environment(p.row)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    # print("List of directions: 0=up; 1=right; 2=down; 3=left")

    while True:
        state, reward, done, info = env.action(env.select_random_action())
        print(state)

        if env.done == True:
            print("Game Over!")
            break
    
    sys.exit(app.exec_())