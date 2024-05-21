application_name = 'Named Entity Recognition'
# pyqt packages
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor

import sys
import torch
from transformers import DistilBertTokenizerFast
import pickle
import matplotlib.pyplot as plt

from qt_main import Ui_Application
from main import BidirectionalMap, compute_device, inference
from distilBERT import DistilBERT


def get_colors(num_colors):
    colormap = plt.get_cmap('viridis')
    # don't use the yellowish color
    skip = 4
    colors = [colormap(i * (256//(num_colors+skip))) for i in range(num_colors+skip)]
    colors = colors[:-skip]
    
    qcolors = [QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255)) for color in colors]
    return qcolors


            
class QT_Action(Ui_Application, QMainWindow):
    
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.processed = False
        self.model = None
        self.tokenizer = None
        with open('label_ind_map.pkl', 'rb') as f:
            self.label_ind_map = pickle.load(f)
        
        # load color
        self.colors = get_colors(len(self.label_ind_map))
        
        # load the model
        self.load_model_action()
        
        
                
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.textEdit_text.textChanged.connect(self.text_edit_action)
        self.textEdit_text.cursorPositionChanged.connect(self.cursor_action)
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
                
        
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'DistilBERT':
            self.model = DistilBERT(nclasses=len(self.label_ind_map)).model
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('dslim/distilbert-NER')
        
        self.model.load_state_dict(torch.load(f'{type(self.model).__name__}_finetuned.pth'))
            
        # move model to GPU
        self.model = self.model.to(compute_device())
    
    
    
    def text_edit_action(self):
        self.processed = False
        
        
    # highlight the given text with color
    def highlightText(self, color, start, end):
        cursor = self.textEdit_text.textCursor()
        format = QTextCharFormat()
        format.setBackground(color)
        cursor.beginEditBlock()
        # define the start and end positions
        cursor.setPosition(start)
        cursor.movePosition(QTextCursor.NextWord, QTextCursor.KeepAnchor, end-start)
        # merge blocks
        cursor.mergeCharFormat(format)
        cursor.endEditBlock()
        
    
    # process the input text, highlight the text based on model prediction
    def process_action(self):
        # get the input sentence
        data = self.textEdit_text.toPlainText()
        
        tok = self.tokenizer(data)
        
        # model inference
        output = inference(self.model, torch.tensor(tok['input_ids']))
        
        # Print the token to characters mapping
        for i, ids in enumerate(output):
            pos = tok.token_to_chars(i)
            if pos is None:
                continue
            start, end = pos
            color = self.colors[ids]
            self.highlightText(color, start, end)
        
        self.processed = True
        
    
    # mouse cursor clicked on textEdit_text
    # display the label in lineEdit_class
    def cursor_action(self):
        if self.processed == False:
            return

        cursor = self.textEdit_text.textCursor()
        cursor.select(QTextCursor.WordUnderCursor)
        format = cursor.charFormat()
        background_color = format.background().color()
        
        if background_color in self.colors:
            color = self.colors.index(background_color)
            label = self.label_ind_map.get_key(color)
            if label == 'O':
                label = 'Others'
            self.lineEdit_class.setText(label)
        

def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()