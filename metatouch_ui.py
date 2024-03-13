#!/usr/bin/env python3
# ==============================================================================
"""
    __  ___     __       ______                 __      
   /  |/  /__  / /_____ /_  __/___  __  _______/ /_ 
  / /|_/ / _ \/ __/ __ `// / / __ \/ / / / ___/ __ \
 / /  / /  __/ /_/ /_/ // / / /_/ / /_/ / /__/ / / /
/_/  /_/\___/\__/\__,_//_/  \____/\__,_/\___/_/ /_/  V.0.1

Author: Abhipol Vibhatasilpin (abhipol@umich.edu)
Description: Configurable data visualization and collection tool for the 
             MetaTouch research project

"""
# ==============================================================================

# System
import sys
import os
import time
import configparser
import subprocess
import socket
from threading import Event, Thread
from datetime import datetime

# Data processing
import numpy as np
import pandas
from scipy.ndimage import rotate
from collections import deque

# UI tools 
from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtCore  import *
from PyQt5.QtGui   import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg

# Custom
from metatouch_label import ClassLabelWidget, StateLabelWidget

# ==============================================================================
# Read in configuration
config = configparser.ConfigParser()
config.read('config.ini')

HOST = config['NETWORK']['HOST'] 
PORT =  int(config['NETWORK']['PORT'])
CHANNELS = config['DATA']['CHANNELS'][1:-1].split(', ') 
NUM_CHANNELS = len(CHANNELS) 
CLASSES = config['DATA']['CLASSES'][1:-1].split(', ') 
CAPTURE_SIZE = int(config['DATA']['CAPTURE_SIZE'])
BATCH_SIZE = int(config['DATA']['BATCH_SIZE'])
FRAME_LENGTH = int(config['PLOT']['FRAME_LENGTH'])
INDEX_WIDTH = int(config['PLOT']['INDEX_WIDTH'])
COLORMAP = config['PLOT']['COLORMAP']
FPS_TICK_RATE = int(config['PLOT']['FPS_TICK_RATE'])

# ==============================================================================

# Global font configuration
font_family = 'Verdana'
fontsize_normal = 11
fontsize_labels = fontsize_normal
fontsize_footer = fontsize_normal + 8

class MetaTouch(QtWidgets.QMainWindow):
    """ Driver class for the application """
    def __init__(self):
        super(MetaTouch, self).__init__()
        uic.loadUi("metatouch_layout.ui", self)
        self.show()
        self.setWindowTitle("MetaTouch Plotter V.0.1")
        
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.installEventFilter(self)

        # Keep track of internal plot handler and data
        self.titles = []
        self.lineplots = []
        self.spectrograms = []
        self.update_signals = []
        self.state_data = {
            "transition_state" : [],
            "timestamp" : [],
        }
        self.transitions = 0
        self.num_frames = 0
        self.state_index = 0
        self.streaming = False

        self.labels = ClassLabelWidget(CLASSES) 
        self.states = StateLabelWidget(["No Touch", "Touch"])

        self.ds_signals = Signals()
        self.ds_signals.read_stream.connect(self.save_stream)
        self.ds_signals.read_fps.connect(self.add_fps)

        # Set up the FPS counter
        self.fps_label = QtWidgets.QLabel()
        self.fps_label.setText(f"FPS: {self.num_frames}")
        self.fps_label.setAlignment(Qt.AlignRight)

        # Set up the status message at the bottom
        self.footer = QtWidgets.QLabel("MetaTouch")
        self.footer.setWordWrap(True)
        self.footer.setFixedWidth(self.width())
        self.footer.setAlignment(Qt.AlignHCenter)

        # Set up conn_stat status disaply
        self.conn_stat = QtWidgets.QLabel("Not Connected")
        self.conn_stat.setWordWrap(True)
        self.conn_stat.setFixedWidth(self.width())
        self.conn_stat.setAlignment(Qt.AlignLeft)

        self.build()

    def build(self):
        """ Called after class constructor """

        # Add encapsulating widgets
        self.LinePane = QtWidgets.QWidget()
        self.LinePaneHL = QtWidgets.QHBoxLayout()
        self.SpecArray = QtWidgets.QWidget()
        self.SpecArrayHL = QtWidgets.QHBoxLayout()
        self.SpecPane = QtWidgets.QWidget()
        self.SpecPaneVL = QtWidgets.QVBoxLayout()
        self.PlotPane = QtWidgets.QWidget()
        self.PlotPaneVL = QtWidgets.QVBoxLayout() 
        self.LinePane.setLayout(self.LinePaneHL)
        self.SpecArray.setLayout(self.SpecArrayHL)
        self.SpecPane.setLayout(self.SpecPaneVL)
        self.PlotPane.setLayout(self.PlotPaneVL)

        # Enforce global pyqtgraph configuration
        pg.setConfigOption('background', (44, 44, 46))
        
        # Set up the spectrograms and line plots
        for i in range(NUM_CHANNELS):
                
            # Create a temp widget that holds lineplot and label
            lineplot_with_label = QtWidgets.QWidget()
            lineplot_with_label_VL = QtWidgets.QVBoxLayout()
            lineplot_with_label.setLayout(lineplot_with_label_VL)
    
            # Generate a label for the plot title       
            title = QtWidgets.QLabel(self)
            title.setText(CHANNELS[i])

            self.titles.append(title)

            # Instantiate a line plot widget and route the signal
            lineplot = LineplotWidget()
            lineplot.read_collected.connect(lineplot.update)
            
            self.update_signals.append(lineplot.read_collected)
            self.lineplots.append(lineplot)

            lineplot_with_label_VL.addWidget(title, alignment=Qt.AlignCenter)
            lineplot_with_label_VL.addWidget(lineplot)

            self.LinePaneHL.addWidget(lineplot_with_label)

            spectrogram = SpectrogramWidget()
            spectrogram.read_collected.connect(spectrogram.update)
            
            self.update_signals.append(spectrogram.read_collected)
            self.spectrograms.append(spectrogram)

            self.SpecArrayHL.addWidget(spectrogram)
        
        # Create a colorbar widget
        self.SpecBar = pg.GraphicsLayoutWidget()
        self.Cbar = pg.ColorBarItem(values=(0,2),
                                    width=6,
                                    colorMap=COLORMAP,
                                    interactive=False,
                                    orientation='horizontal')
        self.SpecBar.addItem(self.Cbar)

        # Assemble the spectogram pane          
        self.SpecPaneVL.addWidget(self.SpecArray)
        self.SpecPaneVL.addWidget(self.SpecBar)

        # Set up the main display
        self.PlotPaneVL.addWidget(self.LinePane, 3)
        self.PlotPaneVL.addWidget(self.SpecPane, 7)
        self.PlotVL.addWidget(self.PlotPane)
        
        # Set up the console widgets
        self.ConsoleGL.addWidget(self.labels, 1, 1, alignment = Qt.AlignLeft)
        self.ConsoleGL.addWidget(self.states, 1, 1, alignment = Qt.AlignRight)

        # Set up the footer widgets
        self.FooterGL.addWidget(self.footer, 1, 1, 
                                alignment=Qt.AlignHCenter)
        self.FooterGL.addWidget(self.fps_label, 1, 1, 
                                alignment=Qt.AlignRight)
        self.FooterGL.addWidget(self.conn_stat, 1, 1,
                                alignment=Qt.AlignLeft)
       
        self.ds = DataSource(self.update_signals, self.conn_stat,
                             self.ds_signals.read_stream, self.ds_signals.read_fps) 
        
        self.socket_thread = self.ds.thread()
        self.socket_thread.start()

        # Set up timers
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.ds.read_channels)
        self.plot_timer.start(1)

        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(FPS_TICK_RATE * 1000)

        # Apply theme
        self.set_appearance()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.TouchBegin:
            self.on_touch()
            if self.streaming:
                self.footer.setText("We have a touch begin")
                self.state_data["transition_state"].append(1)
                self.state_data["timestamp"].append(time.time())
                self.transitions += 1
                self.states.select_element("Touch")
            return True
        elif event.type() == QEvent.TouchEnd:
            if self.streaming:
                self.footer.setText("We have a touch end")
                self.state_data["transition_state"].append(0)
                self.state_data["timestamp"].append(time.time())
                self.transitions += 1
                self.states.select_element("No Touch")
            return True

        return super(MetaTouch, self).eventFilter(obj, event)

    def on_touch(self):
        if not self.streaming:
            self.streaming = True
            self.labels.deactivate()
            self.states.activate()
            self.footer.setText("Continuous Capture")

    def keyPressEvent(self, event):
        """Listen and handle keyboard input."""
        self.footer.setText("MetaTouch")
        # Q
        if event.key()==Qt.Key_Q:
            self.on_q()
        # P
        elif event.key()==Qt.Key_P:
            self.on_p()
        # C 
        elif event.key()==Qt.Key_C:
            self.on_c()
        # S 
        elif event.key()==Qt.Key_S:
            self.on_s()

        if not self.streaming:
            # SpaceBar
            if event.key()==Qt.Key_Space:
                self.on_spacebar()
            # Backspace 
            elif event.key()==Qt.Key_Backspace:
                self.on_backspace()
            # Key Up
            elif event.key()==Qt.Key_Up:
                self.on_up()
            # Key Down
            elif event.key()==Qt.Key_Down:
                self.on_down()
            # Key Left
            elif event.key()==Qt.Key_Left:
                self.on_up()
            # Key Right
            elif event.key()==Qt.Key_Right:
                self.on_down()
        else:
            self.footer.setText("Invalid Keyboard Input.")
    
    def on_q(self):
        """ Q for quit """
        df = pandas.DataFrame(self.state_data)
        df.to_csv("transitions.csv", sep='\t')  
        self.ds.socket.close()
        self.ds.kill_socket.set()
        sys.exit()

    def on_p(self):
        """ P for print screen """
        screenshot = self.PlotPane.grab(self.PlotPane.rect())
        filename = datetime.now().strftime("%Y_%m_%d-%I_%M_%S") 
        screenshot.save(filename + ".png")
        self.footer.setText("Printed to " + filename)
    
    def on_c(self):
        """ C for clear plots """
        for i in range(NUM_CHANNELS):
            self.spectrograms[i].read_collected.emit(np.zeros((FRAME_LENGTH,INDEX_WIDTH)))

    def on_s(self):
        """ S for switch mode """
        self.streaming = not self.streaming
        self.labels.toggle()
        self.states.toggle()
    
        if self.streaming:
            self.footer.setText("Continuous Capture")
        else:
            self.footer.setText("Single Capture")

    def on_up(self):
        """ Up or Left Arrow to move label up """
        if not self.streaming:
            self.labels.move_up()

    def on_down(self):
        """ Down or Right Arrow to move label down """
        if not self.streaming:
            self.labels.move_down()

    def on_spacebar(self):
        """ Spacebar to collect data """
        current_label = self.labels.get_current_label_raw_text()
        current_label = current_label.lower().strip().replace(" ", "_")
        num_frame = self.labels.get_current_frames()
        filename = f"training_data_{current_label}_{num_frame}.npy"
        collected_data = np.empty((NUM_CHANNELS, CAPTURE_SIZE, INDEX_WIDTH)) 
        for i in range(NUM_CHANNELS):
                collected_data[i] = self.spectrograms[i].img_array[-CAPTURE_SIZE:]
        np.save(filename, collected_data)
        self.labels.add_frames_current_label(CAPTURE_SIZE)
        self.footer.setText(f"Collected {CAPTURE_SIZE} frames.")
    
    def on_backspace(self):
        """ Backspace to delete """ 
        current_label = self.labels.get_current_label_raw_text()
        current_label = current_label.lower().strip().replace(" ", "_")
        num_frame = self.labels.get_current_frames() - CAPTURE_SIZE
        filename = f"training_data_{current_label}_{num_frame}.npy"
        if os.path.exists(filename):
            os.remove(filename)
        self.labels.add_frames_current_label(-CAPTURE_SIZE)
        self.footer.setText(f"Deleted {CAPTURE_SIZE} frames.")

    def save_stream(self, batch):
        if self.streaming:
            np.save(f"state_data_batch_{self.state_index}_{time.time()}.npy", batch)
            self.states.add_frames_current_label(BATCH_SIZE)
            self.state_index += BATCH_SIZE 

    def add_fps(self, tick):
        self.num_frames += tick

    def update_fps(self, *args):
        """Update FPS label."""
        fps = int(self.num_frames/FPS_TICK_RATE)
        self.fps_label.setText("FPS: {}".format(fps))
        self.num_frames = 0

    def set_appearance(self):
        self.centralwidget.setContentsMargins(20, 10, 20, 10)
        self.SpecBar.setMaximumHeight(50)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(10, 10, 10))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, Qt.black)
        palette.setColor(QPalette.AlternateBase, Qt.gray)
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, Qt.black)
        palette.setColor(QPalette.Background, QColor(28, 28, 30))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        board_stylesheet = 'background-color: rgb(44, 44, 46); border-radius: 8px'
        self.LinePane.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.LinePane.setStyleSheet(board_stylesheet)
        self.SpecPane.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.SpecPane.setStyleSheet(board_stylesheet)

        self.footer.setFont(QFont(font_family, fontsize_footer))
        
        for title in self.titles:
            title.setContentsMargins(20,0,0,0)
            title.setFont(QFont(font_family, fontsize_normal))
            title.setStyleSheet("color: white; font: bold")
        for spectrogram in self.spectrograms:
            spectrogram.setBackground((44, 44, 46))
        for lineplot in self.lineplots:
            lineplot.setBackground((44, 44, 46))

    def closeEvent(self,e):
        df = pandas.DataFrame(self.state_data)
        df.to_csv("transitions.csv", sep='\t')  
        e.accept()

class DataSource():
    """ Class that handles incoming data """ 

    def __init__(self,signal,message,export_data, export_fps):
        self.signal = signal
        self.message = message
        self.export_data = export_data
        self.export_fps = export_fps
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.kill_socket = Event()
        self.slice = np.zeros((NUM_CHANNELS, INDEX_WIDTH))
        self.queue = deque()
        self.queue.append(self.slice)
        self.batch = []
        self.batch_index = 0

    def read_channels(self):
        for i in range(NUM_CHANNELS):
            self.signal[2*i].emit(self.queue[-1][i])
            self.signal[2*i + 1].emit(self.queue[-1][i])
       
        if self.batch_index < BATCH_SIZE:
            self.batch.append(self.slice)
            self.batch_index += 1
        else:
            self.export_data.emit(self.queue[-1])
            self.batch = []
            self.batch_index = 0

        if len(self.queue) > 1:
            self.queue.popleft()

    def thread(self):
        return Thread(target=self.stream)

    def run_conn_stat(self, conn):
        conn.settimeout(3)
        self.frames = np.zeros((5,NUM_CHANNELS,INDEX_WIDTH))
        while not self.kill_socket.is_set():
            try:          
                if self.kill_socket.is_set(): 
                    break
                signal = np.array([])
                bytes_received = 0
                while(bytes_received < 2 * 4008):
                    message = conn.recv(2 * 4008 - bytes_received)
                    temp = np.frombuffer(message, dtype=np.uint8)
                    signal = np.hstack((signal, temp))
                    bytes_received += temp.shape[0]
                signal = np.asarray(signal, dtype='<B').view(np.uint16)
                signal = np.reshape(signal, (4, 1002))[:,:-2].astype(np.float32) / 4095 * 3.3
                self.frames = np.vstack((self.frames, np.expand_dims(signal, axis=0)))
                self.frames = np.delete(self.frames, 0, axis=0)
                self.slice = np.mean(self.frames, axis=0)
                self.queue.append(self.slice)
                self.export_fps.emit(1) 

            except socket.timeout:
                self.socket.settimeout(10)
                self.message.setText("timeout")
    
    def stream(self):
        self.socket.bind((HOST, PORT))
        self.message.setText("Can not resolve hostname") 
        self.socket.listen(5) 
        self.socket.setblocking(0)
        self.socket.settimeout(20)
        while not self.kill_socket.is_set():
            try:
                conn, addr = self.socket.accept()
                self.message.setText(f"Connected to {addr}")
                self.run_conn_stat(conn)
            except socket.timeout:
                self.message.setText("Ended Connection")
                exit()

class Signals(QObject):
    read_stream = QtCore.pyqtSignal(np.ndarray)
    read_fps = QtCore.pyqtSignal(int)

class SpectrogramWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(SpectrogramWidget, self).__init__()

        self.img = pg.ImageItem()
        self.addItem(self.img)
        
        self.capture_marker = pg.InfiniteLine(pos=FRAME_LENGTH-CAPTURE_SIZE,
                                              angle=0,
                                              movable=False,
                                              bounds=(0,INDEX_WIDTH))
        self.addItem(self.capture_marker)


        self.img_array = np.zeros((FRAME_LENGTH, INDEX_WIDTH))
        self.img.setImage(self.img_array)
        cmap = pg.colormap.get(COLORMAP)

        self.img.setColorMap(colorMap=cmap)

        self.setXRange(0,INDEX_WIDTH)
        self.setYRange(0,FRAME_LENGTH)
        self.setLabel('left', 'Frame')
        self.setLabel('bottom', 'Index')
        self.setMouseEnabled(x=False,y=False)
        self.setMenuEnabled(enableMenu=False)
        self.getPlotItem().hideButtons()
        self.show()

    def update(self, layer):
        if layer.shape[0] == FRAME_LENGTH:
            self.img_array = layer
        else:
            self.img_array = np.roll(self.img_array, -1, 0)
            self.img_array[-1:] = layer
            img_display = np.flip(self.img_array, axis=1)
            img_display = rotate(img_display,angle=90)
            self.img.setImage(img_display)

class LineplotWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(LineplotWidget, self).__init__()
        self.line = pg.PlotDataItem(np.zeros(INDEX_WIDTH))
        self.line.setPen(width=3)
        self.addItem(self.line)
        self.setLabel('left', 'Voltage', units='V')
        self.setLabel('bottom', 'Index')
        self.setMouseEnabled(x=False,y=False)
        self.setMenuEnabled(enableMenu=False)
        self.show()

    def update(self, layer):
        self.line.setData(layer)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mt = MetaTouch()
    app.setStyle("Fusion")
    app.setFont(QFont(font_family, fontsize_normal))
    sys.exit(app.exec())        
