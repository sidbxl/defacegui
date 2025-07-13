import sys
from PyQt5.QtWidgets import QApplication
from deface.deface_gui import DefaceGUIMain

def main():
    app = QApplication(sys.argv)
    window = DefaceGUIMain()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
