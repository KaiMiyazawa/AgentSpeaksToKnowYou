from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLineEdit, QLabel, QDialog
from PyQt5.QtCore import Qt, pyqtSignal



#gender = input('性別を入力してください。男:"Male" 女:"Female" ')
##['Employed' 'Homemaker' 'Student' nan 'Retired' 'Unable to work']
#employment = input('職業を入力してください。働いている:"Employed" 主婦:"Homemaker" 学生:"Student" 退職:"Retired" 就労不能:"Unable to work" その他:"nan" ')
##20-29' '30-39' '40-49' '-19' '50-59' '60-69']
#age = input('年齢を入力してください。: 19歳以下:"-19" 20代:"20-29" 30代:"30-39" 40代:"40-49" 50代:"50-59" 60代以上:"60-69" ')

class InputWindowGender(QDialog):
	info_submitted = pyqtSignal(str)

	def __init__(self):
		super().__init__()

		self.setWindowTitle("Input Window")
		self.setGeometry(150, 150, 300, 200)

		self.layout = QVBoxLayout()

		self.label = QLabel("性別を選択してください:", self)
		self.layout.addWidget(self.label)

		button_layout = QHBoxLayout()

		self.male_button = QPushButton("男", self)
		self.male_button.clicked.connect(lambda: self.submit_info("Male"))
		button_layout.addWidget(self.male_button)

		self.female_button = QPushButton("女", self)
		self.female_button.clicked.connect(lambda: self.submit_info("FeMale"))
		button_layout.addWidget(self.female_button)

		self.layout.addLayout(button_layout)

		self.setLayout(self.layout)

	def submit_info(self, info):
		self.info_submitted.emit(info)
		self.close()











class InputWindowAge(QDialog):
	info_submitted = pyqtSignal(str)

	def __init__(self):
		super().__init__()

		self.setWindowTitle("Input Window")
		self.setGeometry(150, 150, 300, 200)

		self.layout = QVBoxLayout()

		self.label = QLabel("年齢を選択してください:", self)
		self.layout.addWidget(self.label)

		button_layout = QHBoxLayout()

		self.age1_button = QPushButton("19歳以下", self)
		self.age1_button.clicked.connect(lambda: self.submit_info("-19"))
		button_layout.addWidget(self.age1_button)

		self.age2_button = QPushButton("20代", self)
		self.age2_button.clicked.connect(lambda: self.submit_info("20-29"))
		button_layout.addWidget(self.age2_button)

		self.age3_button = QPushButton("30代", self)
		self.age3_button.clicked.connect(lambda: self.submit_info("30-39"))
		button_layout.addWidget(self.age3_button)

		self.age4_button = QPushButton("40代", self)
		self.age4_button.clicked.connect(lambda: self.submit_info("40-49"))
		button_layout.addWidget(self.age4_button)

		self.age5_button = QPushButton("50代", self)
		self.age5_button.clicked.connect(lambda: self.submit_info("50-59"))
		button_layout.addWidget(self.age5_button)

		self.age6_button = QPushButton("60代以上", self)
		self.age6_button.clicked.connect(lambda: self.submit_info("60-69"))
		button_layout.addWidget(self.age6_button)

		self.layout.addLayout(button_layout)

		self.setLayout(self.layout)

	def submit_info(self, info):
		self.info_submitted.emit(info)
		self.close()



class InputWindowEmployment(QDialog):
	info_submitted = pyqtSignal(str)

	def __init__(self):
		super().__init__()

		self.setWindowTitle("Input Window")
		self.setGeometry(150, 150, 300, 200)

		self.layout = QVBoxLayout()

		self.label = QLabel("職業を選択してください:", self)
		self.layout.addWidget(self.label)

		button_layout = QHBoxLayout()

		self.employment3_button = QPushButton("学生", self)
		self.employment3_button.clicked.connect(lambda: self.submit_info("Student"))
		button_layout.addWidget(self.employment3_button)

		self.employment1_button = QPushButton("働いている", self)
		self.employment1_button.clicked.connect(lambda: self.submit_info("Employed"))
		button_layout.addWidget(self.employment1_button)

		self.employment2_button = QPushButton("主婦", self)
		self.employment2_button.clicked.connect(lambda: self.submit_info("Homemaker"))
		button_layout.addWidget(self.employment2_button)

		self.employment4_button = QPushButton("退職", self)
		self.employment4_button.clicked.connect(lambda: self.submit_info("Retired"))
		button_layout.addWidget(self.employment4_button)

		self.employment5_button = QPushButton("就労不能", self)
		self.employment5_button.clicked.connect(lambda: self.submit_info("Unable to work"))
		button_layout.addWidget(self.employment5_button)

		self.employment6_button = QPushButton("その他", self)
		self.employment6_button.clicked.connect(lambda: self.submit_info("nan"))
		button_layout.addWidget(self.employment6_button)

		self.layout.addLayout(button_layout)

		self.setLayout(self.layout)

	def submit_info(self, info):
		self.info_submitted.emit(info)
		self.close()