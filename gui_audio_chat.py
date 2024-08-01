# 初めての人が見てもわかるように、丁寧で簡潔なコメントアウトを書くように心がけてください。

# 必要なライブラリをインポート
import sys
from openai import OpenAI
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt, QTimer, QThread
import os
from pydub import AudioSegment
import time
import threading
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import pyaudio
import wave

# モジュールをインポート
import list_var
from InputWindow import InputWindowGender, InputWindowAge, InputWindowEmployment, DisplayResultWindow

global index
index = -1

# OpenAI APIを使用するためのクライアントを作成
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# メッセージのリストを作成
msg = [{"role": "system", "content": "ユーザーのパーソナリティを引き出すような会話をするシステムです。ユーザーが話している話題について、相手が興味があると思われることについて質問してください。簡潔な質問を返答するようにしてください。質問を考えるにあたって、ユーザーのパーソナリティがわかるような質問をしてください。"},
	{"role": "user", "content": "こんにちは。何か話題はありますか？"},]

# 音声録音クラス
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

class AudioRecorder(QThread):
	# クラスの初期化
	def __init__(self):
		super().__init__()
		self.frames = []
		self.is_recording = False
		self.progress_symbol = ["｜", "／", "ー", "＼"]
		self.progress_index = 0

	# 音声録音を開始する
	def run(self):
		self.is_recording = True
		audio = pyaudio.PyAudio()
		stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
		self.frames = []
		while self.is_recording:
			sound_data = stream.read(CHUNK, exception_on_overflow=False)
			self.frames.append(sound_data)

		stream.stop_stream()
		stream.close()
		audio.terminate()

		with wave.open(f"./user_audios/{index}.wav", 'wb') as waveFile:
			waveFile.setnchannels(CHANNELS)
			waveFile.setsampwidth(audio.get_sample_size(FORMAT))
			waveFile.setframerate(RATE)
			waveFile.writeframes(b''.join(self.frames))

	# 音声録音を停止する
	def stop(self):
		self.is_recording = False

#class ResultWindow(QMainWindow):
#	def __init__(self):
#		super().__init__()
#		self.initUI()

#	def initUI(self):
#		self.setWindowTitle('性格診断結果')
#		self.setGeometry(100, 100, 800, 600)

#		central_widget = QWidget()
#		self.setCentralWidget(central_widget)

#		layout = QVBoxLayout()
#		central_widget.setLayout(layout)

#		self.result_area = QTextEdit()
#		self.result_area.setReadOnly(True)
#		layout.addWidget(self.result_area)

#		self.result_area.append("性格診断結果")

# ChatWindowクラス
class ChatWindow(QMainWindow):

	# クラスの初期化
	def __init__(self):
		super().__init__()
		self.initUI()
		self.audio_recorder = AudioRecorder()
		self.created_files = []
		self.user_wav_file = []
		self.user_text_file = []

	# UIの初期化
	def initUI(self):
		self.setWindowTitle('AIと会話して、おすすめの文芸の一節を教えてもらおう！')
		self.setGeometry(100, 100, 800, 600)

		central_widget = QWidget()
		self.setCentralWidget(central_widget)

		layout = QVBoxLayout()
		central_widget.setLayout(layout)

		self.chat_area = QTextEdit()
		self.chat_area.setReadOnly(True)
		layout.addWidget(self.chat_area)

		input_layout = QHBoxLayout()

		self.record_button = QPushButton('録音開始')
		self.record_button.clicked.connect(self.toggle_recording)
		input_layout.addWidget(self.record_button)

		self.finish_button = QPushButton('性格診断')
		self.finish_button.clicked.connect(self.finish)
		input_layout.addWidget(self.finish_button)
		self.finish_button.setEnabled(False)

		layout.addLayout(input_layout)

	# 録音ボタンのトグル
	def toggle_recording(self):
		global index
		if not self.audio_recorder.is_recording:
			index += 1
			self.audio_recorder.start()
			self.record_button.setText("録音終了")
			self.finish_button.setEnabled(False)
			self.chat_area.append("Recording started...")
			self.start_progress_indicator()
		else:
			self.chat_area.append("Recording stopped...")
			self.timer.stop()
			self.audio_recorder.stop()
			self.created_files.append(f"./user_audios/{index}.wav")
			self.user_wav_file.append(f"./user_audios/{index}.wav")
			self.record_button.setText("録音開始")
			self.finish_button.setEnabled(True)
			self.chat_area.append("")
			time.sleep(0.5)
			self.finish_button.setEnabled(False)
			audio_file = open(f"./user_audios/{index}.wav", "rb")
			transcription = client.audio.transcriptions.create(
				model="whisper-1",
				file=audio_file,
				language="ja"
			)
			transcription_text_file = f"./user_texts/{index}.txt"
			with open(transcription_text_file, "w", encoding='utf-8') as f:
				f.write(transcription.text)
			self.created_files.append(transcription_text_file)
			self.user_text_file.append(transcription_text_file)
			user_message = transcription.text
			if user_message:
				self.chat_area.append(f"You: {user_message}")
				self.process_message(user_message)
			self.finish_button.setEnabled(True)

	# システム音声を再生する
	def play_system_audio(self):
		system_audio_file_path = f"./system_audios/{index}.wav"
		system_audio_file_path_mp3 = f"./system_audios/{index}.mp3"
		self.created_files.append(system_audio_file_path_mp3)
		self.created_files.append(system_audio_file_path)
		with client.audio.speech.with_streaming_response.create(
			model='tts-1',
			voice='alloy',
			input=self.system_responce,
		) as response:
			response.stream_to_file(system_audio_file_path_mp3)
		audio_mp3 = AudioSegment.from_mp3(system_audio_file_path_mp3)
		audio_mp3.export(system_audio_file_path, format="wav")

		# silent
		silent_file_path = f"./system_audios/silent.wav"
		silent_audio = AudioSegment.silent(duration=len(audio_mp3))
		silent_audio.export(silent_file_path, format="wav")
		self.created_files.append(silent_file_path)
		self.user_wav_file.append(silent_file_path)

		os.system(f"afplay {system_audio_file_path}")
		# wait for the audio to finish playing

	# メッセージを処理する
	def process_message(self, message):
		try:
			msg.append({"role": "user", "content": message})
			response = client.chat.completions.create(
				model="gpt-3.5-turbo",
				messages=msg
			)
			assistant_message = response.choices[0].message.content
			msg.append({"role": "assistant", "content": assistant_message})
			with open(f"./system_texts/{index}.txt", "w", encoding='utf-8') as f:
				f.write(message)
			self.created_files.append(f"./system_texts/{index}.txt")
			self.chat_area.append(f"ChatGPT: {assistant_message}\n")
			self.system_responce = assistant_message
			play_audio_thread = threading.Thread(target=self.play_system_audio)
			play_audio_thread.daemon = True
			play_audio_thread.start()
		except Exception as e:
			self.chat_area.append(f"エラーが発生しました: {str(e)}\n")

	# プログレスインジケータを更新する
	def update_progress_indicator(self):
		self.chat_area.moveCursor(self.chat_area.textCursor().Start)
		self.chat_area.moveCursor(self.chat_area.textCursor().End)
		cursor = self.chat_area.textCursor()
		cursor.movePosition(QTextCursor.End)
		cursor.deletePreviousChar()
		cursor.insertText(self.audio_recorder.progress_symbol[self.audio_recorder.progress_index])
		self.audio_recorder.progress_index = (self.audio_recorder.progress_index + 1) % 4

	# プログレスインジケータを開始する
	def start_progress_indicator(self):
		self.timer = QTimer()
		self.chat_area.append(' ')
		self.timer.timeout.connect(self.update_progress_indicator)
		self.timer.start(100)

	# 性別、年齢、職業を受け取る
	def receive_gender_info(self, info):
		self.gender = info

	def receive_age_info(self, info):
		self.age = info

	def receive_employment_info(self, info):
		self.employment = info
		if info == 'nan':
			self.employment = np.nan

	# プログラムの終了と同時に、性格診断結果を表示する
	def finish(self):
		self.record_button.setEnabled(False)
		self.finish_button.setEnabled(False)

		# concatenate user_wav_file ========================
		combined = AudioSegment.empty()
		combined_file_path = f"./user_audios/combined.wav"
		for file in self.user_wav_file:
			combined += AudioSegment.from_wav(file)

		combined.export(combined_file_path, format="wav")
		self.created_files.append(combined_file_path)
		# ==================================================

		# concatenate user_text_file ========================
		combined_text = ""
		combined_text_file_path = f"./user_texts/combined.txt"

		for file in self.user_text_file:
			with open(file, "r", encoding='utf-8') as f:
				combined_text += f.read() + "[SEP]"
		combined_text = combined_text[:-5]

		with open(combined_text_file_path, "w", encoding='utf-8') as f:
			f.write(combined_text)
		self.created_files.append(combined_text_file_path)
		# ==================================================

		# apply openSMILE ==================================
		from opensmile import Smile, FeatureSet, FeatureLevel
		from scipy.io import wavfile
		import numpy as np
		smile = Smile(
			feature_set=FeatureSet.eGeMAPSv02,
			feature_level=FeatureLevel.Functionals,
		)

		# 性別、年齢、職業を入力する
		self.input_window = InputWindowGender()
		self.input_window.info_submitted.connect(self.receive_gender_info)
		self.input_window.exec_()

		self.input_window = InputWindowAge()
		self.input_window.info_submitted.connect(self.receive_age_info)
		self.input_window.exec_()

		self.input_window = InputWindowEmployment()
		self.input_window.info_submitted.connect(self.receive_employment_info)
		self.input_window.exec_()

		# openSMILEを使って音声データを処理する
		fs, x = wavfile.read(combined_file_path)
		x = x / np.iinfo(x.dtype).max
		result = smile.process_signal(x, fs)
		wav_df = result.reset_index()

		wav_df['gender'] = self.gender
		wav_df['employment'] = self.employment
		wav_df['age'] = self.age
		wav_df = wav_df.reindex(list_var.wav_columns_list, axis=1)
		wav_df.drop('start', axis=1, inplace=True)
		wav_df.drop('end', axis=1, inplace=True)
		wav_df.drop('id', axis=1, inplace=True)

		file_path_wav_csv = f"./csvs/wav.csv"
		wav_df.to_csv(file_path_wav_csv, index=False, encoding='utf-8')
		# ==================================================

		# apply NLP to CSV =================================
		import pandas as pd
		text_df = pd.DataFrame([combined_text], columns=['text'])
		text_df['gender'] = self.gender
		text_df['employment'] = self.employment
		text_df['age'] = self.age
		text_df = text_df.reindex(list_var.text_columns_list, axis=1)

		file_path_text_csv = f"./csvs/text.csv"
		text_df.to_csv(file_path_text_csv, index=False, encoding='utf-8')
		# ==================================================

		# make dataframe for personality prediction with NLP ======
		text_df['utterances'] = text_df['text'].apply(lambda x: x.split('[SEP]'))

		text_df['speech_count'] = text_df['utterances'].apply(len)
		text_df['average_speech_length'] = text_df['utterances'].apply(lambda x: sum(len(utterance) for utterance in x) / len(x))

		text_df_const = text_df.copy()

		text_df_const = text_df_const.drop(columns=["text", "utterances", "speech_count"])
		new_column_order = ["interlocutor_id", "average_speech_length", "gender", "age", "employment"]
		text_df_const = text_df_const[new_column_order]

		text_df_bi = text_df_const.copy()

		text_df_bi["dummy_19"] = text_df_bi["age"].apply(lambda x: 1 if x == "-19" else 0)
		text_df_bi["dummy_gender"] = text_df_bi["gender"].apply(lambda x: 1 if x == "male" else 0)
		text_df_bi["dummy_student"] = text_df_bi["employment"].apply(lambda x: 1 if x == "Student" else 0)

		text_df_group = text_df_bi.groupby('interlocutor_id').agg({
			'average_speech_length': 'mean',
			'gender': 'first',
			'age': 'first',
			'employment': 'first'
		}).reset_index()

		categorical = ['gender', 'age', 'employment']

		# MeCabを使って形態素解析を行う
		import MeCab
		from collections import Counter

		os.environ['MECABRC'] = '/opt/homebrew/etc/mecabrc'
		mecab = MeCab.Tagger("-d /opt/homebrew/lib/mecab/dic/unidic")

		def analyze_pos(texts):
			pos_counter = Counter()

			for sentence in texts:
				# 形態素解析を実行
				node = mecab.parseToNode(sentence)
				while node:
					pos = node.feature.split(',')[0]
					if pos not in ['BOS/EOS']:
						pos_counter[pos] += 1
					node = node.next
			return pos_counter

		def calculate_pos_ratios(pos_counter, total_count):
			pos_ratios = {pos: count / total_count for pos, count in pos_counter.items()}
			return pos_ratios

		text_df_bi['text'] = text_df['utterances']
		text_df_bi['pos_counts'] = text_df_bi['text'].apply(analyze_pos)
		text_df_bi['pos_ratios'] = text_df_bi['pos_counts'].apply(lambda x: calculate_pos_ratios(x, sum(x.values())))
		text_df_bi.drop(["pos_counts"], axis=1, inplace=True)

		text_df_pos_ratio = pd.json_normalize(text_df_bi['pos_ratios'])
		text_df_bi = pd.concat([text_df_bi, text_df_pos_ratio], axis=1)
		text_df_bi = text_df_bi.drop(columns=['pos_ratios'])

		text_df_bi = text_df_bi.drop(columns=['interlocutor_id'])
		text_df_bi = text_df_bi.drop(columns=['text'])

		text_columns_list = ['average_speech_length', '副詞', '接頭辞', '動詞', '助動詞', '補助記号', '名詞', '助詞', '形容詞', '形状詞', '代名詞', '感動詞', '連体詞', '接尾辞', '記号', '接続詞', '空白']
		#この形式のデータフレームを作成する
		text_df_bi = text_df_bi.reindex(columns=text_columns_list, fill_value=0)
		# null値を0に変換

		text_columns_list += ['gender_Female', 'gender_Male', 'age_-19', 'age_20-29', 'age_30-39', 'age_40-49', 'age_50-59', 'age_60-69', 'employment_Employed', 'employment_Homemaker', 'employment_Retired', 'employment_Student', 'employment_Unable to work']

		fillFalse_list = ['gender_Female', 'gender_Male', 'age_-19', 'age_20-29', 'age_30-39', 'age_40-49', 'age_50-59', 'age_60-69', 'employment_Employed', 'employment_Homemaker', 'employment_Retired', 'employment_Student', 'employment_Unable to work']
		text_df_bi[fillFalse_list] = False

		# 適当にカラムの補完をする
		if self.gender == 'Male':
			text_df_bi['gender_Male'] = True
		else:
			text_df_bi['gender_Female'] = True

		if self.age == '-19':
			text_df_bi['age_-19'] = True
		elif self.age == '20-29':
			text_df_bi['age_20-29'] = True
		elif self.age == '30-39':
			text_df_bi['age_30-39'] = True
		elif self.age == '40-49':
			text_df_bi['age_40-49'] = True
		elif self.age == '50-59':
			text_df_bi['age_50-59'] = True
		else:
			text_df_bi['age_60-69'] = True

		if self.employment == 'Employed':
			text_df_bi['employment_Employed'] = True
		elif self.employment == 'Homemaker':
			text_df_bi['employment_Homemaker'] = True
		elif self.employment == 'Retired':
			text_df_bi['employment_Retired'] = True
		elif self.employment == 'Student':
			text_df_bi['employment_Student'] = True
		else:
			text_df_bi['employment_Unable to work'] = True

		x = text_df_bi
		# ==================================================

		# predict personality with NLP ======================

		# predict personality NLP_DF TO PERSONALITY==============

		import lightgbm as lgb
		print('\n')
		self.chat_area.append("\n性格診断結果\n")

		# Openness
		model_openness = lgb.Booster(model_file='models/model_openness.txt')
		y_pred_openness = model_openness.predict(x, num_iteration=model_openness.best_iteration)
		print("openness", y_pred_openness)
		self.chat_area.append(f"Openness: {y_pred_openness}")

		# Conscientiousness
		model_conscientiousness = lgb.Booster(model_file='models/model_conscientiousness.txt')
		y_pred_conscientiousness = model_conscientiousness.predict(x, num_iteration=model_conscientiousness.best_iteration)
		print("conscientiousness", y_pred_conscientiousness)
		self.chat_area.append(f"Conscientiousness: {y_pred_conscientiousness}")

		# Extraversion
		model_exteaversion = lgb.Booster(model_file="models/model_extraversion.txt")
		y_pred_exteaversion = model_exteaversion.predict(x, num_iteration=model_exteaversion.best_iteration)
		print("exteaversion", y_pred_exteaversion)
		self.chat_area.append(f"Extraversion: {y_pred_exteaversion}")

		# Agreeableness
		model_agreeableness = lgb.Booster(model_file='models/model_agreeableness.txt')
		y_pred_agreeableness = model_agreeableness.predict(x, num_iteration=model_agreeableness.best_iteration)
		print("agreeableness", y_pred_agreeableness)
		self.chat_area.append(f"Agreeableness: {y_pred_agreeableness}")

		# Neuroticism
		model_neuroticism = lgb.Booster(model_file='models/model_neuroticism.txt')
		y_pred_neuroticism = model_neuroticism.predict(x, num_iteration=model_neuroticism.best_iteration)
		print("neuroticism", y_pred_neuroticism)
		self.chat_area.append(f"Neuroticism: {y_pred_neuroticism}")

		# =========================================================

		print('\n')
		self.chat_area.append("\n音声診断結果\n")

		# predict personality WAV_DF TO PERSONALITY==============

		x = wav_df.drop(columns=['employment', 'age'])
		x = pd.get_dummies(x, columns=['gender'])

		if self.gender == 'Male':
			x['gender_Female'] = False
		else:
			x['gender_Male'] = False

		# Factor1 (好悪)

		model_factor1 = lgb.Booster(model_file='models/model_audio1.txt')
		y_pred_factor1 = model_factor1.predict(x, num_iteration=model_factor1.best_iteration)
		print("factor1", y_pred_factor1)
		self.chat_area.append(f"Factor1: {y_pred_factor1}")

		# Factor2 (上手さ)

		model_factor2 = lgb.Booster(model_file='models/model_audio2.txt')
		y_pred_factor2 = model_factor2.predict(x, num_iteration=model_factor2.best_iteration)
		print("factor2", y_pred_factor2)
		self.chat_area.append(f"Factor2: {y_pred_factor2}")

		# Factor3 (活動性)

		model_factor3 = lgb.Booster(model_file='models/model_audio3.txt')
		y_pred_factor3 = model_factor3.predict(x, num_iteration=model_factor3.best_iteration)
		print("factor3", y_pred_factor3)
		self.chat_area.append(f"Factor3: {y_pred_factor3}")

		# Factor4 (速さ感)

		model_factor4 = lgb.Booster(model_file='models/model_audio4.txt')
		y_pred_factor4 = model_factor4.predict(x, num_iteration=model_factor4.best_iteration)
		print("factor4", y_pred_factor4)
		self.chat_area.append(f"Factor4: {y_pred_factor4}")

		# Factor5 (スタイル)

		model_factor5 = lgb.Booster(model_file='models/model_audio5.txt')
		y_pred_factor5 = model_factor5.predict(x, num_iteration=model_factor5.best_iteration)
		print("factor5", y_pred_factor5)
		self.chat_area.append(f"Factor5: {y_pred_factor5}")

		print('\n')
		# =========================================================

		# 文章を出す =============================================
		# アンケートデータの整形============================================================================================================
		# アンケートのデータ読み込み
		df_questionnaire = pd.read_csv("性格特性と文章の捉え方についての調査_2024年7月22日_03.26_masked.csv", header=0)

		df_result = df_questionnaire.drop(df_questionnaire.index[0:2])
		df_result = df_result.drop(df_questionnaire.columns[0:12], axis=1)

		df_int = df_result.astype(int)
		df_int["Openness"] = [y_pred_openness] * len(df_int)
		df_int["Conscientiousness"] = [y_pred_conscientiousness] * len(df_int)
		df_int["Extraversion"] = [y_pred_exteaversion] * len(df_int)
		df_int["Agreeableness"] = [y_pred_agreeableness] * len(df_int)
		df_int["Neuroticism"] = [y_pred_neuroticism] * len(df_int)

		test_factor5 = [y_pred_factor1, y_pred_factor2, y_pred_factor3, y_pred_factor4, y_pred_factor5]

		df_sentences = pd.read_csv("性格診断_出力(小説) - シート1.csv", header=None)
		df_sentences.columns = ["文章", "作者"]

		# テキストの方のデータを比較===============================================================================================

		# 予測結果をリストに変換
		# 予測結果をリストに変換（最初の要素をスカラーとして抽出）
		test_bigfive = [
			y_pred_openness[0],
			y_pred_conscientiousness[0],
			y_pred_exteaversion[0],
			y_pred_agreeableness[0],
			y_pred_neuroticism[0]
		]

		text_columns = [col for col in df_int.columns if col.endswith("_6")]
		text_columns.remove("Q3.1_6")
		text_columns.extend(["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"])

		df_symp = df_int[text_columns]

		# 最後の5列の値を取り出して辞書を作成
		last_5_columns = df_symp.columns[-5:]
		result_dict = {index: row[last_5_columns].tolist() for index, row in df_symp.iterrows()}

		# コサイン類似度を計算
		cos_sim_list = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
		cos_sim = cosine_similarity([test_bigfive], df_symp[cos_sim_list])[0]

		# 類似度が最も高い３行を取得
		top3_indices = cos_sim.argsort()[-3:][::-1]
		top3_rows = df_symp.iloc[top3_indices]

		# '_6'で終わる列の平均を計算
		columns_ending_with_6 = [col for col in df_symp.columns if col.endswith('_6')]
		mean_values = top3_rows[columns_ending_with_6].mean()

		# 最も平均値が高い列名を取得
		max_mean_column = mean_values.idxmax()

		# 正規表現を使用してすべての数値部分を抽出
		matches = re.findall(r'\d+', max_mean_column)
		if matches:
			# リストから数値部分を選択
			most_similar_question_number = int(matches[-2])
		else:
			print("No number found in the string")

		sentence_text = df_sentences.iloc[most_similar_question_number - 1, 0]
		author_text = df_sentences.iloc[most_similar_question_number - 1, 1]

		#print(f"sentence: {sentence_text}, author: {author_text}")

		# factor5 to text =================================================

		test_factor5 = [y_pred_factor1[0], y_pred_factor2[0], y_pred_factor3[0], y_pred_factor4[0], y_pred_factor5[0]]

		required_columns = [col for col in df_int.columns if col.startswith('Q5.')]

		df_factors = df_int[required_columns]

		unrequired_columns = [col for col in df_factors if col.endswith("_6")]

		df_factors = df_factors.drop(unrequired_columns, axis=1)

		# 文章ごと、factor5ごとに指標の平均値を算出する
		factors_means = df_factors.mean()

		means_dict = {}

		for index, value in factors_means.items():
			key = index.rsplit('_', 1)[0]
			if key not in means_dict:
				means_dict[key] = []
			means_dict[key].append(value)

		# コサイン類似度を計算し、最も類似している番号を見つける
		max_similarity = -1
		most_similar_key = None

		# reshape to 2D array for cosine_similarity function
		test_vector = np.array(test_factor5).reshape(1, -1)

		for key, values in means_dict.items():
			values_vector = np.array(values).reshape(1, -1)
			similarity = cosine_similarity(test_vector, values_vector)[0][0]
			if similarity > max_similarity:
				max_similarity = similarity
				most_similar_key = key

		# 正規表現を使用してすべての数値部分を抽出
		matches = re.findall(r'\d+', most_similar_key)
		if matches:
			# リストから最後の数値部分を選択
			most_similar_question_number = int(matches[-1])
		else:
			print("No number found in the string")

		# 文章と作者の抽出
		sentence_voice = df_sentences.iloc[most_similar_question_number - 1, 0]
		author_voice = df_sentences.iloc[most_similar_question_number - 1, 1]

		#print(f"sentence: {sentence_voice}, author: {author_voice}")

		self.chat_area.append("\n言語的特徴による性格診断結果\n")
		self.chat_area.append(f"文章: {sentence_text}, 作者: {author_text}")
		self.chat_area.append("\n音声的特徴による性格診断結果\n")
		self.chat_area.append(f"文章: {sentence_voice}, 作者: {author_voice}")

		# =========================================================

		# 結果を表示する
		self.result_window = DisplayResultWindow(sentence_text, author_text, sentence_voice, author_voice)
		self.result_window.show()



	# ウィンドウを閉じるときに、録音を停止し、作成されたファイルを削除する
	def closeEvent(self, event):
		self.audio_recorder.stop()

		# remove all created files
		created_file_dir = ["./user_audios", "./user_texts", "./system_audios", "./system_texts", "./csvs"]

		for dir in created_file_dir:
			for file in os.listdir(dir):
				file_path = os.path.join(dir, file)
				try:
					os.remove(file_path)
				except Exception as e:
					print(f"Error while deleting file", str(e))

		event.accept()

# メイン関数
if __name__ == '__main__':
	app = QApplication(sys.argv)
	chat_window = ChatWindow()
	chat_window.show()
	sys.exit(app.exec_())
