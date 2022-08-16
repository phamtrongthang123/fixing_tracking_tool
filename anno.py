from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
import sys
from PIL.ImageQt import ImageQt
from PIL import Image
from PyQt5.Qt import Qt
from pathlib import Path 
from visualize import *
import numpy as np 
import cv2
from PyQt5 import QtCore
import os 
import bz2 

class Window(QWidget):
    def __init__(self, max_person=-1, path='B5KplpNzL-A'):
        super().__init__()
        self.max_person=max_person
        self.path = path
        self.checkpoint = Path(path)/'checkpoint'
        self.count_updating = 0
        self.track_data, self.inv_track = self.get_tracking(self.path)
        self.acceptDrops()
        # set the title
        self.setWindowTitle("Image")
 
        # setting  the geometry of window
        self.setGeometry(0, 0, 1500, 800)

        self.hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        # creating label
        self.label = QLabel()
        self.max_length = len(list(Path(f'frames/{self.path}').glob("*.png")))
        self.plot_qim()
 
        self.label.resize(self.pixmap.width(),
                          self.pixmap.height())

        self.hbox.addWidget(self.label)
        self.label_frame = QLabel(f'Frame: {self.index} \nUpdate times: {self.count_updating}')
        vbox.addWidget(self.label_frame)
        self.hbox.addLayout(vbox)

        self.grid = QGridLayout()
        self.create_class_ids()

        vbox.addLayout(self.grid)
        self.button = QPushButton('Update')
        self.button.clicked.connect(self.button_clicked)
        self.label_updated = QLabel(f'')
        vbox.addWidget(self.button)
        vbox.addWidget(self.label_updated)
        self.setLayout(self.hbox)
        self.setFocus()
        self.show()
            
    def button_clicked(self):
        self.modify()
        

    def get_tracking(self, path):
        results = {}
        inv_results = {}
        path = Path(path)
        fname = path.name
        
        track_path = path/f'tracking_results/{fname}_anno.txt'
        if not os.path.exists(track_path):
            track_path = path/f'tracking_results/{fname}.txt'
            self.index = 1
        else: 
            with open(f'{path}/resume_index.txt', 'r') as f: 
                self.index,self.count_updating = [int(x) for x in f.readline().strip().split(',')]
        with open(track_path, 'r') as f:
            all_lines = [x.strip() for x in f.readlines()]
        
        # read lines: normal and inverse index 
        for line in all_lines:
            frame_id, cls_id, x,y,w,h, conf,_,_,_ = line.split(',')
            cls_id = int(cls_id)
            x,y,w,h,conf = [float(i) for i in (x,y,w,h,conf)]
            x0,y0,x1,y1 = x,y,x+w,y+h
            # normal 
            if frame_id in results:
                results[frame_id]['class_ids'].append(cls_id)
                results[frame_id]['bboxes'].append((x0,y0,x1,y1))
                results[frame_id]['conf'].append(conf)
            else:
                results[frame_id] = {}
                results[frame_id]['class_ids'] = [cls_id]
                results[frame_id]['bboxes'] = [(x0,y0,x1,y1)]
                results[frame_id]['conf'] = [conf]
            # inverse 
            if cls_id in inv_results:
                inv_results[cls_id].append(frame_id)
            else:
                inv_results[cls_id] = [frame_id]

    
        # max person 
        self.max_index = 0
        for k,v in results.items():
            if self.max_person == -1:
                self.max_person = max(self.max_person, len(v['class_ids']))
            self.max_index = max(self.max_index, max(v['class_ids']))
        return results, inv_results

    def save_tracking(self):
        path = self.path
        results = {}
        path = Path(path)
        fname = path.name
        track_path = path/f'tracking_results/{fname}_anno.txt'
        with open(track_path, 'w') as f:
            for key in  sorted(self.track_data):
                cls_ids = self.track_data[str(key)]['class_ids']
                bboxes = self.track_data[str(key)]['bboxes']
                confs = self.track_data[str(key)]['conf']
                for cl, bb, conf in zip(cls_ids, bboxes, confs):
                    x0,y0,x1,y1 = bb 
                    w = x1-x0
                    h = y1-y0 
                    f.write(f'{key},{cl},{x0},{y0},{w},{h},{conf},-1,-1,-1\n')
        # print(results)
        with open(f'{path}/resume_index.txt', 'w') as f: 
            f.write(f'{self.index},{self.count_updating}')
        return results

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.forward()
            # self.save_tracking()
        elif event.key() == Qt.Key_Left:
            self.backward()
            # self.save_tracking()
        elif event.key() == Qt.Key_Q:
            self.modify()
            self.close()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.modify()            
            self.forward()
        elif event.key() == Qt.Key_U:
            self.undo()

    def undo(self):
        if self.count_updating == 0:
            return
        self.count_updating -= 1
        path = Path(self.path)
        fname=path.name 
        track_path = path/f'tracking_results/{fname}_anno.txt'
        resume = path/'resume_index.txt'
        checkpoint_index = self.checkpoint/f'{self.count_updating}'

        anno_out = checkpoint_index/"anno.bz2"
        resume_out = checkpoint_index/"resume.bz2"

        with bz2.open(anno_out, mode="rb") as fin, open(track_path, "wb") as fout:
            fout.write(fin.read())
        with bz2.open(resume_out, mode="rb") as fin, open(resume, "wb") as fout:
            fout.write(fin.read())
        old_index = self.index
        self.track_data, self.inv_track = self.get_tracking(self.path)
        self.index = old_index
        self.label_updated.setText(f'UNDO AT STEP {self.count_updating}!')
        self.plot_qim()
        self.setFocus()


    def mousePressEvent(self, event):
        self.setFocus()

    def modify(self):
        # check overlap 
        labels, texts =[], [] 
        for i in range(0, self.grid.count(), 2):
            label, text = self.grid.itemAt(i).widget().text(), self.grid.itemAt(i+1).widget().text()
            label, text = int(label), int(text)
            if text in texts:
                self.label_updated.setText(f"CAN'T UPDATE AT FRAME {self.index}! \nTHERE ARE OVERLAPPING!")
                return
            labels.append(label)
            texts.append(text) 
        # update
        new_cls = []
        spread = []
        # this list will have n-1 as the frame_id, and n-th is the target text 
        ordr = {k:[[],[]] for k in range(1, self.max_index+1)}
        for i in range(0, self.grid.count(), 2):
            label, text = self.grid.itemAt(i).widget().text(), self.grid.itemAt(i+1).widget().text()
            label, text = int(label), int(text)
            # update the sequence id for max_person 
            # if label > self.max_person and label != text:
            #     for frame_id in self.inv_track[label]:
            #         tmp = []
            #         for clsid in self.track_data[str(frame_id)]['class_ids']:
            #             if clsid == label: 
            #                 tmp.append(text)
            #             else:
            #                 tmp.append(clsid)
            #         self.track_data[str(frame_id)]['class_ids'] = tmp 
            #     self.inv_track[text] += self.inv_track[label]
            #     self.inv_track[text] = list(sorted(self.inv_track[text]))
            #     del self.inv_track[label]

            # # update sequence for consecutive
            # else:                
            if label != text:
                # self.index already has the other lines taking care of
                for frame_id in range(self.index+1, self.max_length):
                    count = 0
                    # we don't want to trouble ourselves with different length 
                    if str(frame_id) not in self.track_data: break
                    if len(self.track_data[str(frame_id)]['class_ids']) != len(self.track_data[str(self.index)]['class_ids']): break
                    for i,clsid in enumerate(self.track_data[str(frame_id)]['class_ids']):
                        if clsid == label: 
                            ordr[label][0].append(frame_id)
                            ordr[label][1].append(i)
                            count += 1
                    if count == 0: break                         
                ordr[label][0].append(text)
            new_cls.append(text)
        self.track_data[str(self.index)]['class_ids'] = new_cls

        # after the loop, we have this new ordr list to know what to change and the min index.
        # We want to avoid update too much. 
        # We may actually need try catch here, but let's stay with if for now
        min_frameid = 99999
        for k,v in ordr.items():
            if len(v[0]) > 1:
                min_frameid = min(min_frameid, v[0][-2])
        # start updating
        # print(min_frameid)
        if min_frameid != 99999:
            for frame_id in range(self.index+1, min_frameid+1):
                tmp = self.track_data[str(frame_id)]['class_ids']
                for k,v in ordr.items():
                    if len(v[0]) > 1: 
                        target = v[0][-1]
                        idx = v[1][frame_id-self.index-1]
                        tmp[idx] = target 
                    
                # if duplicate exists
                if len(tmp) != len(set(tmp)):
                    # print("break?")
                    break 
                self.track_data[str(frame_id)]['class_ids'] = tmp
        
        

        self.label_updated.setText(f'UPDATED AT FRAME {self.index}!')
        self.plot_qim()
        self.setFocus()
        # checkpoint before save new stuff
        if self.count_updating != 0:
            self.store_checkpoint()
        else:
            self.count_updating += 1
        self.save_tracking()

    def store_checkpoint(self):
        path = Path(self.path)
        fname=path.name 
        track_path = path/f'tracking_results/{fname}_anno.txt'
        resume = path/'resume_index.txt'
        checkpoint_index = self.checkpoint/f'{self.count_updating}'
        os.makedirs(checkpoint_index, exist_ok=True)

        anno_out = checkpoint_index/"anno.bz2"
        resume_out = checkpoint_index/"resume.bz2"

        with open(track_path, mode="rb") as fin, bz2.open(anno_out, "wb") as fout:
            fout.write(fin.read())
        with open(resume, mode="rb") as fin, bz2.open(resume_out, "wb") as fout:
            fout.write(fin.read())
        self.count_updating += 1

    def forward(self):
        if self.index < self.max_length:
            self.index +=1
        self.plot_qim()
        self.set_new_frame_text()
        self.create_class_ids()
    def backward(self):
        if self.index != 1:
            self.index -=1
        self.plot_qim()
        self.set_new_frame_text()
        self.create_class_ids()

    def plot_qim(self):
        im = Image.open(f'frames/{self.path}/{str(self.index).zfill(10)}.png').convert("RGB")
        im = self.vis_img(im, self.index)
        data = im.tobytes("raw","RGB")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qim)
        self.label.setPixmap(self.pixmap)

    def create_class_ids(self):
        # don't use while, the delete happens later (static) not right away so it will stuck at delete 0 if you hope if will reduce like list python
        for i in range(self.grid.count()):
            self.grid.itemAt(i).widget().deleteLater()
        if str(self.index) in self.track_data:
            cls_ids = self.track_data[str(self.index)]['class_ids']
            for i,cls in enumerate(cls_ids):
                clslabel = QLabel(f'{cls}')
                clsedit2 = QLineEdit(f'{cls}')
                self.grid.addWidget(clslabel, i+1, 0)
                self.grid.addWidget(clsedit2, i+1, 1)


    def set_new_frame_text(self):
        self.label_frame.setText(f'Frame: {self.index} \nUpdate times: {self.count_updating}')

    def vis_img(self, im, index):
        if str(index) not in self.track_data:
            return im 
        open_cv_image = np.array(im) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        boxes = self.track_data[str(index)]['bboxes']
        cls_ids = self.track_data[str(index)]['class_ids']
        scores = self.track_data[str(index)]['conf']
        drawed_img = vis(open_cv_image, boxes, scores, cls_ids)
        drawed_img = drawed_img[:,:,::-1].copy()
        return Image.fromarray(drawed_img)
# create pyqt5 app
App = QApplication(sys.argv)
 
# create the instance of our Window
window = Window(path=sys.argv[1], max_person=int(sys.argv[2]))
window.show()
# start the app
sys.exit(App.exec())